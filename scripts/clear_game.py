import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

game = sys.argv[1] if len(sys.argv) > 1 else "658"
from api.db import get_connection
conn = get_connection()
n = conn.execute(f"DELETE FROM draws WHERE game = '{game}'").rowcount
conn.close()
print(f"Deleted {n} rows for game={game}")
