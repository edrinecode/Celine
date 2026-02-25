from datetime import datetime


class ClinicalTools:
    @staticmethod
    def vitals_risk_heuristic(text: str) -> str:
        red_flags = [
            "chest pain",
            "shortness of breath",
            "fainting",
            "stroke",
            "seizure",
            "suicidal",
            "bleeding",
        ]
        lowered = text.lower()
        hits = [flag for flag in red_flags if flag in lowered]
        if hits:
            return f"Potential high-risk symptoms detected: {', '.join(hits)}"
        return "No explicit red-flag symptom keywords detected."

    @staticmethod
    def timestamp_tool() -> str:
        return datetime.utcnow().isoformat() + "Z"
