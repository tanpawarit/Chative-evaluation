"""Seed goldens and optional synthesis helpers for the intent agent eval."""

from dataclasses import dataclass
from typing import List

@dataclass
class IntentGolden:
    """Single test scenario for intent extraction."""

    name: str
    input: str
    intent: str

    def to_dict(self):
        return {
            "name": self.name,
            "input": self.input,
            "intent": self.intent,
        }


SEED_GOLDENS: List[IntentGolden] = [
    IntentGolden(
        name="gpu_price_cap",
        input="หาการ์ดจอ RTX 4070 ราคาไม่เกิน 25,000 บาท พร้อมส่ง",
        intent="inquire_product",
    ),
    IntentGolden(
        name="greeting",
        input="สวัสดีครับ มีใครอยู่ไหม",
        intent="greet",
    ),
    IntentGolden(
        name="check_iphone_stock",
        input="มี iPhone 15 Pro Max สี Natural Titanium ไหมครับ",
        intent="check_stock",
    ),
    IntentGolden(
        name="claim_warranty_monitor",
        input="จอเปิดไม่ติด ซื้อมาเมื่อวาน เคลมได้ไหม",
        intent="warranty_claim",
    ),
    IntentGolden(
        name="tech_support_gpu_crash",
        input="เล่นเกมแล้วเด้งหลุด ขึ้น Error 0x887A0005 ครับ",
        intent="technical_support",
    ),
    IntentGolden(
        name="tech_support_error_query",
        input="ถ้า Error 0x887A0005 เกิดขึ้นระหว่างแข่งทัวร์นาเมนต์สำคัญ ผลลัพธ์จะเป็นอย่างไร?",
        intent="technical_support",
    ),
]
