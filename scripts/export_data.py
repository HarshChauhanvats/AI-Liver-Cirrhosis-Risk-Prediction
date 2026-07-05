import sqlite3
import pandas as pd

conn = sqlite3.connect('hospital.db')

df = pd.read_sql_query(
    "SELECT * FROM patients",
    conn
)

df.to_csv(
    'patients_export.csv',
    index=False
)

conn.close()

print("CSV exported successfully!")