import re


banned_keywords = ["rm -rf", "shutdown", "format"]


def filter_text(self, text: str) -> str:
    """
    对文本型输入做清理、过滤、长度截断。
    """
    if not isinstance(text, str):
        raise ValueError("输入必须是字符串")

    cleaned = re.sub(r"\s+", " ", text.strip())

    if len(cleaned) > self.max_length:
        cleaned = cleaned[:self.max_length]
        print(f"输入被截断到最大长度 {self.max_length} 字符")

    for keyword in banned_keywords:
        if keyword in cleaned.lower():
            raise ValueError(f"检测到非法关键词：{keyword}")

    return cleaned
