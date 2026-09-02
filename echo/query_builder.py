from argus_logger import get_recent_anomalies, get_all_anomalies

ATTACK_DESCRIPTIONS = {
    "micro_structuring": (
        "Micro-Structuring — the transaction variance dropped to exactly 0.0, "
        "indicating identical recurring transaction amounts, a common pattern "
        "used to evade AML threshold rules."
    ),
    "automated_bot_activity": (
        "Automated Bot Activity — near-zero variance combined with an elevated "
        "Risk Score, suggesting scripted transactions or card testing behavior."
    ),
    "account_takeover": (
        "Account Takeover (ATO) — the anomaly score is significantly above "
        "the normal baseline, indicating transaction patterns that deviate sharply "
        "from the user's historical behavior."
    ),
    "suspicious_transfer": (
        "Suspicious Transfer — Anomaly score is above the detection threshold, "
        "suggesting a potentially illicit transfer or layering activity."
    ),
    "unknown": (
        "an anomaly of unclassified type — insufficient signal characteristics "
        "to determine attack vector."
    )
}

def build_query_from_anomaly(anomaly: dict) -> str:
    attack_desc = ATTACK_DESCRIPTIONS.get(
        anomaly.get("attack_hint", "unknown"),
        ATTACK_DESCRIPTIONS["unknown"]
    )

    query = f"""
An anomaly has been detected by the Argus Fin AI system in a payment gateway 
transaction stream. Here are the technical details:

- Primary Feature/Entity: {anomaly.get('primary_feature', anomaly.get('sensor_id', 'unknown'))}
- Dataset/Merchant: {anomaly.get('entity_id', anomaly.get('facility', 'unknown'))}
- Timestamp/Step: {anomaly['timestamp']}
- MAE Anomaly Score: {anomaly['mse_score']} 
- Variance: {anomaly['variance']}
- Detected Pattern: {attack_desc}

Based on financial fraud literature and known AML/CFT attack patterns:
1. What fraud technique does this pattern most closely match?
2. What is the recommended immediate analyst response?
3. What downstream accounts or merchants should be cross-checked?
4. What is the potential financial consequence if this goes unaddressed?
""".strip()

    return query

def build_query_from_latest() -> tuple[dict, str]:
    recent = get_recent_anomalies(n=1)
    if not recent:
        return None, None
    anomaly = recent[-1]
    return anomaly, build_query_from_anomaly(anomaly)

def build_query_from_id(anomaly_id: str) -> tuple[dict, str]:
    all_logs = get_all_anomalies()
    match = next((a for a in all_logs if a["id"] == anomaly_id), None)
    if not match:
        return None, None
    return match, build_query_from_anomaly(match)

def build_summary_query(n: int = 5) -> str:
    recent = get_recent_anomalies(n=n)
    if not recent:
        return None

    lines = []
    for a in recent:
        lines.append(
            f"- Entity {a.get('primary_feature', a.get('sensor_id', 'unknown'))} | Merchant: {a.get('entity_id', a.get('facility', 'unknown'))} | "
            f"MAE: {a['mse_score']} | Pattern: {a['attack_hint']} | "
            f"Time: {a['timestamp']}"
        )

    query = f"""
The Argus Fin AI system has detected {len(recent)} anomalies recently 
in the payment gateway stream. Summary:

{chr(10).join(lines)}

Based on this pattern of detections:
1. Is there evidence of a coordinated multi-account fraud ring?
2. Which merchant or entity appears most at risk?
3. What fraud campaign does this pattern suggest?
4. What is the recommended security posture for the risk team?
""".strip()

    return query