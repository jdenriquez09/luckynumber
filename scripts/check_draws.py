import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from api.db import get_connection

game = sys.argv[1] if len(sys.argv) > 1 else "658"
conn = get_connection(read_only=True)
df = conn.execute(
    "SELECT draw_date, weekday, n1,n2,n3,n4,n5,n6, jackpot_php "
    "FROM draws WHERE game=? ORDER BY draw_date DESC LIMIT 5",
    [game]
).fetchdf()
conn.close()
print(df.to_string())
