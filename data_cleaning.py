import pandas as pd
import numpy as np

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # ── 1. Standardise column names ──────────────────────────
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_'))

    # ── 2. Rename to standard names ──────────────────────────
    rename_map = {
        'name':          'car_name',
        'year':          'year',
        'selling_price': 'selling_price',
        'km_driven':     'km_driven',
        'fuel':          'fuel_type',
        'seller_type':   'seller_type',
        'transmission':  'transmission',
        'owner':         'owner',
        'mileage':       'mileage',
        'engine':        'engine_cc',
        'max_power':     'max_power_bhp',
        'torque':        'torque',
        'seats':         'seats'
    }
    df = df.rename(columns={k: v
                             for k, v in rename_map.items()
                             if k in df.columns})

    # ── 3. Extract brand from car name ───────────────────────
    if 'car_name' in df.columns:
        df['brand'] = df['car_name'].str.split().str[0].str.strip()
        df['model'] = (df['car_name']
                         .str.split()
                         .str[1:3]
                         .str.join(' ')
                         .str.strip())

    # ── 4. Clean numeric columns ─────────────────────────────
    if 'mileage' in df.columns:
        df['mileage'] = (df['mileage']
                           .astype(str)
                           .str.extract(r'([\d.]+)')[0]
                           .astype(float))

    if 'engine_cc' in df.columns:
        df['engine_cc'] = (df['engine_cc']
                             .astype(str)
                             .str.extract(r'([\d.]+)')[0]
                             .astype(float))

    if 'max_power_bhp' in df.columns:
        df['max_power_bhp'] = (df['max_power_bhp']
                                 .astype(str)
                                 .str.extract(r'([\d.]+)')[0]
                                 .astype(float))

    # ── 5. Drop torque (inconsistent units) ──────────────────
    if 'torque' in df.columns:
        df = df.drop(columns=['torque'])

    # ── 6. Remove outliers ───────────────────────────────────
    df = df[df['selling_price'] > 50000]
    df = df[df['selling_price'] < 10000000]
    df = df[df['km_driven']     > 100]
    df = df[df['km_driven']     < 500000]

    # ── 7. Drop duplicates ───────────────────────────────────
    df = df.drop_duplicates()

    # ── 8. Feature engineering ───────────────────────────────
    current_year    = 2026
    df['age']       = current_year - df['year']
    df['age']       = df['age'].clip(lower=0)

    df['km_per_year'] = (df['km_driven'] /
                          df['age'].replace(0, 0.5))

    df['age_km_interaction'] = df['age'] * df['km_driven']

    df['is_low_mileage'] = (
        df['km_per_year'] < 5000).astype(int)

    brand_tier = {
        'Maruti':     'Budget',
        'Datsun':     'Budget',
        'Renault':    'Budget',
        'Hyundai':    'Mid',
        'Honda':      'Mid',
        'Tata':       'Mid',
        'Kia':        'Mid',
        'Nissan':     'Mid',
        'Ford':       'Mid',
        'Volkswagen': 'Premium',
        'Skoda':      'Premium',
        'Toyota':     'Premium',
        'MG':         'Premium',
        'Jeep':       'Premium',
        'BMW':        'Luxury',
        'Mercedes':   'Luxury',
        'Audi':       'Luxury',
        'Volvo':      'Luxury',
        'Jaguar':     'Luxury',
        'Land':       'Luxury',
    }
    df['brand_tier'] = (df['brand']
                          .map(brand_tier)
                          .fillna('Mid'))

    owner_map = {
        'First Owner':          1,
        'Second Owner':         2,
        'Third Owner':          3,
        'Fourth & Above Owner': 4,
        'Test Drive Car':       1,
    }
    if 'owner' in df.columns:
        df['owner_num'] = (df['owner']
                             .map(owner_map)
                             .fillna(2))

    # ── 9. Drop rows with nulls in key columns ────────────────
    key_cols = ['selling_price', 'km_driven',
                'year', 'fuel_type', 'transmission']
    df = df.dropna(subset=key_cols)

    # ── 10. Fill remaining nulls ─────────────────────────────
    num_cols = df.select_dtypes(
                   include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(
                       df[num_cols].median())

    cat_cols = df.select_dtypes(
                   include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(
                       df[cat_cols].mode().iloc[0])

    df = df.reset_index(drop=True)
    return df


def get_cleaning_report(raw_df, clean_df):
    report = {
        'raw_rows':         len(raw_df),
        'clean_rows':       len(clean_df),
        'rows_removed':     len(raw_df) - len(clean_df),
        'removal_pct':      round(
            (len(raw_df) - len(clean_df)) /
             len(raw_df) * 100, 2),
        'raw_nulls':        int(raw_df.isnull().sum().sum()),
        'clean_nulls':      int(clean_df.isnull().sum().sum()),
        'duplicate_removed': int(
            raw_df.duplicated().sum()),
        'features_engineered': [
            'age', 'km_per_year',
            'age_km_interaction',
            'is_low_mileage', 'brand_tier',
            'owner_num', 'brand', 'model'
        ]
    }
    return report