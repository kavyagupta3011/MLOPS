import sqlite3
con = sqlite3.connect("mlflow.db")
con.execute("UPDATE alembic_version SET version_num = ?", ("0584bdc529eb",))
con.commit()
con.close()
print("Fixed.")
