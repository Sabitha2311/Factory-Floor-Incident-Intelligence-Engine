import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_incidents(num_records=1000, output_path="data/incident_reports.csv"):
    np.random.seed(42)
    random.seed(42)

    # Base Machine IDs
    machine_ids = ["002c2969-c559-4f02-85a2-f2403051e800", "M-101", "M-102", "PUMP-20", "CNV-05", "ROBOT-A"]
    
    incident_types = ["Mechanical", "Electrical", "Safety", "Quality", "Process"]
    incident_probs = [0.4, 0.25, 0.05, 0.1, 0.2] # Imbalanced types
    
    severities = ["Low", "Medium", "High", "Critical"]
    severity_probs = [0.5, 0.3, 0.15, 0.05] # Imbalanced severities

    templates = {
        "Mechanical": [
            "Loud {adjective} noise coming from {component}. Seems like {symptom}.",
            "Found {symptom} on the {component}. Unit is {adjective}.",
            "{component} failed due to {symptom}.",
            "Operator reported {symptom} near {component}."
        ],
        "Electrical": [
            "{component} is shorting out. {symptom} observed.",
            "Voltage drop on {component}. System is {adjective}.",
            "Tripped breaker at {component}. {symptom}.",
            "{symptom} from the main {component} panel."
        ],
        "Safety": [
            "Safety guard removed from {component}. {adjective} hazard.",
            "Operator bypassed safety interlock on {component}. {symptom}.",
            "Spill near {component} causing {symptom}."
        ],
        "Quality": [
            "Product defect rate spiked at {component}. Parts are {adjective}.",
            "{symptom} causing off-spec materials from {component}."
        ],
        "Process": [
            "Process bottleneck at {component}. {symptom} slowing down line.",
            "{component} cycle time increased due to {symptom}. {adjective} flow."
        ]
    }
    
    vocab = {
        "Mechanical": {
            "component": ["bearing", "motor", "conveyor belt", "gearbox", "hydraulic pump", "shaft", "valve"],
            "symptom": ["wear and tear", "excessive vibration", "overheating", "grinding", "leakage"],
            "adjective": ["unstable", "shaking", "loud", "hot", "broken"]
        },
        "Electrical": {
            "component": ["VFD", "PLC", "contactor", "wiring harness", "sensor", "power supply"],
            "symptom": ["sparking", "flickering", "smoke", "communication loss", "ground fault"],
            "adjective": ["burnt", "unresponsive", "offline", "erratic"]
        },
        "Safety": {
            "component": ["safety gate", "e-stop", "light curtain", "walkway", "chemical tank"],
            "symptom": ["slip hazard", "exposure", "bypassed switch", "trip hazard"],
            "adjective": ["dangerous", "unsafe", "critical", "exposed"]
        },
        "Quality": {
            "component": ["inspection camera", "packaging station", "molding dye", "extruder"],
            "symptom": ["misalignment", "color deviation", "dimension out of spec", "surface scratch"],
            "adjective": ["rejected", "non-compliant", "substandard"]
        },
        "Process": {
            "component": ["feed hopper", "assembly robot", "cooling tower", "mixer"],
            "symptom": ["jamming", "material starvation", "pressure drop", "overflow"],
            "adjective": ["slow", "halted", "inefficient"]
        }
    }

    # Root Causes mapped to symptoms
    root_cause_map = {
        "wear and tear": "Lack of lubrication",
        "excessive vibration": "Unbalanced load",
        "overheating": "Cooling system failure",
        "sparking": "Frayed wire",
        "smoke": "Short circuit",
        "slip hazard": "Oil leak",
        "misalignment": "Sensor calibration drift",
        "jamming": "Foreign object debris"
    }

    data = []
    start_date = datetime(2025, 1, 1)

    for _ in range(num_records):
        inc_type = np.random.choice(incident_types, p=incident_probs)
        severity = np.random.choice(severities, p=severity_probs)
        machine = np.random.choice(machine_ids)
        
        # Select vocab
        comp = random.choice(vocab[inc_type]["component"])
        symp = random.choice(vocab[inc_type]["symptom"])
        adj = random.choice(vocab[inc_type]["adjective"])
        
        template = random.choice(templates[inc_type])
        text = template.format(component=comp, symptom=symp, adjective=adj)
        
        # Add some random variations to text
        if random.random() > 0.7:
            text = text.lower()
        if random.random() > 0.8:
            text += f" Checked on {machine}."
            
        root_cause = root_cause_map.get(symp, "General equipment degradation")
        duration_mins = int(np.random.exponential(scale=30 if severity in ["Low", "Medium"] else 120))
        
        timestamp = start_date + timedelta(minutes=random.randint(0, 500000))
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "machine_id": machine,
            "incident_text": text,
            "incident_type": inc_type,
            "severity": severity,
            "component": comp,
            "symptom": symp,
            "root_cause": root_cause,
            "downtime_minutes": duration_mins
        })

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_records} synthetic incident records at {output_path}")

if __name__ == "__main__":
    generate_synthetic_incidents(1500)
