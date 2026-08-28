import sqlite3

DB_PATH = "graphshield.db"

def init_db():
    """Initializes SQLite database with dual-hash support for graphs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS figure_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_name TEXT NOT NULL,
            figure_path TEXT NOT NULL,
            phash TEXT NOT NULL,
            dhash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_figure_hash(paper_name: str, figure_path: str, phash: str, dhash: str):
    """Stores figure metadata along with pHash and dHash signatures."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO figure_hashes (paper_name, figure_path, phash, dhash)
        VALUES (?, ?, ?, ?)
    ''', (paper_name, figure_path, phash, dhash))
    conn.commit()
    conn.close()

def calculate_hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Computes bitwise XOR difference between two 64-bit hex hash strings."""
    val1 = int(hash1_hex, 16)
    val2 = int(hash2_hex, 16)
    return bin(val1 ^ val2).count('1')

def find_matches(query_phash: str, query_dhash: str, threshold: int = 12):
    """
    Performs hybrid matching against stored figures using pHash and dHash distances.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT paper_name, figure_path, phash, dhash FROM figure_hashes')
    rows = cursor.fetchall()
    conn.close()

    matches = []
    
    for paper_name, fig_path, stored_phash, stored_dhash in rows:
        p_dist = calculate_hamming_distance(query_phash, stored_phash)
        d_dist = calculate_hamming_distance(query_dhash, stored_dhash)
        
        # Weighted hybrid score calculation
        avg_distance = (p_dist + d_dist) / 2.0
        
        if avg_distance <= threshold:
            similarity = max(0.0, (1.0 - (avg_distance / 64.0)) * 100.0)
            
            matches.append({
                "matched_paper": paper_name,
                "matched_figure": fig_path,
                "phash_distance": p_dist,
                "dhash_distance": d_dist,
                "hybrid_distance": avg_distance,
                "similarity_score": round(similarity, 2)
            })

    return matches