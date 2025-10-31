# modules/assignment.py
import math
import pandas as pd
from collections import defaultdict

def infer_train_id_from_pkg(pkg_id, trains_df):
    """
    Infer train_id from package_id prefix (first two digits).
    If inference fails, return 'UNKNOWN'.
    """
    try:
        prefix = str(pkg_id)[:2]
        idx = int(prefix)  # 1-based index used when generating: 01 -> train index 1
        if 1 <= idx <= len(trains_df):
            return trains_df.iloc[idx - 1]['train_id']
    except Exception:
        pass
    return "UNKNOWN"

def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    Assign packages to persons minimizing visits using the corrected logic:
    - full capacity persons per warehouse first
    - treat leftovers as atomic and pack them with Best-Fit-Decreasing
    Returns:
        assignments_df: rows (package_id, warehouse_id, train_id, person)
        summary_df: pivot table (train_id x warehouse_id) -> number of distinct persons handling that train/warehouse
        per_train_detail: dict train_id -> DataFrame (Warehouse, Person, package_ids list, count)
        metadata: dict with counts and metrics
    """
    # Defensive copies
    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    # Ensure package ids are strings
    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)

    # infer train_id for each package
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    # Build list of packages per warehouse (preserve order)
    wh_to_pkgs = defaultdict(list)
    for _, row in packages.iterrows():
        wh = row['warehouse_id']
        pid = row['package_id']
        tid = row['train_id']
        wh_to_pkgs[wh].append({'package_id': pid, 'train_id': tid})

    # Total packages
    total_packages = len(packages)
    Hn = math.ceil(total_packages / capacity)
    persons = [f"H{i+1}" for i in range(Hn)]

    # Step A: Full-capacity allocations per warehouse
    assignments = []  # list of dicts: package_id, warehouse_id, train_id, person
    person_idx = 0

    leftovers_per_warehouse = {}  # wh -> list of remaining package dicts

    for wh, pkg_list in wh_to_pkgs.items():
        idx = 0
        n = len(pkg_list)
        f_i = n // capacity
        # assign f_i full persons
        for f in range(f_i):
            if person_idx >= Hn:
                raise RuntimeError("Not enough persons computed; logic error.")
            person = persons[person_idx]
            # assign next capacity packages to this person
            for k in range(capacity):
                pkg = pkg_list[idx]
                assignments.append({
                    'package_id': pkg['package_id'],
                    'warehouse_id': wh,
                    'train_id': pkg['train_id'],
                    'person': person
                })
                idx += 1
            person_idx += 1
        # leftover
        leftover_pkgs = pkg_list[idx:]
        if leftover_pkgs:
            leftovers_per_warehouse[wh] = leftover_pkgs

    # Step B: Assign leftovers (atomic) using Best-Fit-Decreasing
    # Create bins for remaining persons (each bin: person id and used capacity)
    bins = []
    # If person_idx already used some persons, remaining persons will be used for leftovers.
    for i in range(person_idx, Hn):
        bins.append({'person': persons[i], 'used': 0, 'allocs': []})

    # Create list of (warehouse, leftover_count, list_of_pkgs)
    leftover_items = []
    for wh, pkgs in leftovers_per_warehouse.items():
        leftover_items.append((wh, len(pkgs), pkgs))
    # sort descending by leftover size
    leftover_items.sort(key=lambda x: x[1], reverse=True)

    for wh, count, pkgs in leftover_items:
        placed = False
        # Best-Fit: find bin with smallest remaining capacity after placing (but sufficient)
        best_bin = None
        best_after = None
        for b in bins:
            if b['used'] + count <= capacity:
                after = b['used'] + count
                if best_after is None or after < best_after:
                    best_after = after
                    best_bin = b
        if best_bin is not None:
            # place all pkgs to this person
            best_bin['used'] += count
            best_bin['allocs'].append((wh, pkgs))
            placed = True
        else:
            # No existing bin can take it (shouldn't happen because Hn computed), but if it does,
            # create a new bin if possible (fallback) else raise
            if len(bins) < (Hn - person_idx):
                new_person_index = person_idx + len(bins)
                new_person = persons[new_person_index]
                bnew = {'person': new_person, 'used': count, 'allocs': [(wh, pkgs)]}
                bins.append(bnew)
                placed = True
            else:
                # As fallback, split the leftover across persons (only if absolutely necessary)
                # We'll assign greedily across bins with available capacity
                remaining = pkgs.copy()
                for b in bins:
                    avail = capacity - b['used']
                    if avail <= 0:
                        continue
                    take = min(avail, len(remaining))
                    take_pkgs = remaining[:take]
                    b['allocs'].append((wh, take_pkgs))
                    b['used'] += take
                    remaining = remaining[take:]
                    if not remaining:
                        break
                if remaining:
                    raise RuntimeError("Unable to pack leftovers into persons — inconsistent Hn.")

    # Commit bins allocations to assignments
    for b in bins:
        person = b['person']
        for wh, alloc_pkgs in b['allocs']:
            for pkg in alloc_pkgs:
                assignments.append({
                    'package_id': pkg['package_id'],
                    'warehouse_id': wh,
                    'train_id': pkg['train_id'],
                    'person': person
                })

    # Final assignments DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Sanity: ensure every package assigned
    assigned_count = assignments_df['package_id'].nunique()
    if assigned_count != total_packages:
        raise RuntimeError(f"Assigned {assigned_count} packages but expected {total_packages}.")

    # Build summary pivot: for each train x warehouse -> number of distinct persons
    # --- Build summary pivot: train × warehouse ---
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    
    # --- Ensure all warehouses are always shown ---
    all_warehouses = list(warehouses_df["warehouse_id"])
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    
    # --- Convert to fractional persons (packages / capacity) ---
    summary_df = summary_df / capacity
    
    # --- Reset index for display ---
    summary_df = summary_df.reset_index()
    
    # --- Identify numeric warehouse columns only ---
    warehouse_cols = [c for c in summary_df.columns if c.startswith("W")]
    
    # --- Add total persons (ceiling) ---
    summary_df["Total Persons"] = np.ceil(summary_df[warehouse_cols].sum(axis=1)).astype(int)
    
    # --- Round for cleaner display ---
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    # Build per-train detailed mappings
    per_train_detail = {}
    for tid, grp in assignments_df.groupby('train_id'):
        # group by warehouse and person, list package_ids
        detail_rows = []
        for (wh, person), g in grp.groupby(['warehouse_id', 'person']):
            pkgs = list(g['package_id'])
            detail_rows.append({'warehouse': wh, 'person': person, 'packages': pkgs, 'count': len(pkgs)})
        detail_df = pd.DataFrame(detail_rows).sort_values(['warehouse', 'person']).reset_index(drop=True)
        per_train_detail[tid] = detail_df

    metadata = {
        'total_packages': total_packages,
        'capacity': capacity,
        'total_persons': Hn,
        'full_allocations': person_idx,
        'leftover_persons': len(bins),
    }

    return assignments_df, summary_df, per_train_detail, metadata
