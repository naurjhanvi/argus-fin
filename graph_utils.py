import os
import pandas as pd
from neo4j import GraphDatabase

# Configure Neo4j connection from environment variables with sensible defaults
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class GraphDB:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def ingest_transactions(self, df: pd.DataFrame):
        if not self.driver:
            return
            
        # Clean columns if needed
        cols = list(df.columns)
        account_count = 0
        for i, col in enumerate(cols):
            if col.strip() == "Account":
                account_count += 1
                if account_count == 2:
                    cols[i] = "To_Account"
            cols[i] = cols[i].strip()
        df.columns = cols

        with self.driver.session() as session:
            # First create constraints if they don't exist
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
            
            # Write data
            for _, row in df.iterrows():
                # Handles pandas renaming of duplicate columns
                to_acc_col = "To_Account" if "To_Account" in df.columns else "Account.1"
                
                if 'Account' in row and pd.notna(row['Account']) and to_acc_col in row and pd.notna(row[to_acc_col]):
                    session.run(
                        """
                        MERGE (orig:Account {id: $nameOrig})
                        MERGE (dest:Account {id: $nameDest})
                        CREATE (orig)-[:TRANSACTED {
                            amount: $amount, 
                            type: $type, 
                            timestamp: $timestamp,
                            isLaundering: $isLaundering
                        }]->(dest)
                        """,
                        nameOrig=str(row['Account']),
                        nameDest=str(row[to_acc_col]),
                        amount=float(row['Amount Paid']) if 'Amount Paid' in row else 0.0,
                        type=str(row.get('Payment Format', 'UNKNOWN')),
                        timestamp=str(row.get('Timestamp', '')),
                        isLaundering=int(row.get('Is Laundering', 0))
                    )

    def get_fraud_ring(self, target_account_id: str, depth: int = 2):
        if not self.driver:
            return {"nodes": [], "edges": []}
            
        with self.driver.session() as session:
            # Find the neighborhood around the target account up to a certain depth
            query = f"""
            MATCH path = (a:Account {{id: $account_id}})-[*1..{depth}]-(b:Account)
            RETURN path
            LIMIT 100
            """
            result = session.run(query, account_id=target_account_id)
            
            nodes = set()
            edges = []
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nodes.add(node["id"])
                for rel in path.relationships:
                    edges.append({
                        "source": rel.start_node["id"],
                        "target": rel.end_node["id"],
                        "amount": rel.get("amount", 0.0),
                        "type": rel.get("type", "")
                    })
                    
            return {
                "nodes": [{"id": n, "label": n} for n in nodes],
                "edges": edges
            }

graph_db = GraphDB()
