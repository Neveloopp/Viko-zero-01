PAD = 0
BOS = 1
USR = 2
AST = 3
EOS = 4
BYTE_OFFSET = 5
VOCAB_SIZE = 261

SPECIAL_NAMES = {
    PAD: "<PAD>",
    BOS: "<BOS>",
    USR: "<USR>",
    AST: "<AST>",
    EOS: "<EOS>",
}

class ByteTokenizer:
    vocab_size = VOCAB_SIZE

    def encode_text(self, text):
        return [b + BYTE_OFFSET for b in str(text).encode("utf-8")]

    def decode_text(self, ids):
        data = bytearray()
        for token in ids:
            token = int(token)
            if BYTE_OFFSET <= token < VOCAB_SIZE:
                data.append(token - BYTE_OFFSET)
        return data.decode("utf-8", errors="replace")

    def encode_record(self, record):
        if "messages" in record:
            out = [BOS]
            for msg in record["messages"]:
                role = str(msg.get("role", "")).lower()
                text = str(msg.get("content", ""))
                if role in ("user", "usuario"):
                    out.append(USR)
                elif role in ("assistant", "asistente", "viko"):
                    out.append(AST)
                else:
                    continue
                out.extend(self.encode_text(text))
            out.append(EOS)
            return out

        if "prompt" in record and "response" in record:
            return (
                [BOS, USR]
                + self.encode_text(record["prompt"])
                + [AST]
                + self.encode_text(record["response"])
                + [EOS]
            )

        if "text" in record:
            return [BOS] + self.encode_text(record["text"]) + [EOS]

        return []

    def encode_chat_prompt(self, history, user_text, max_tokens=None):
        out = [BOS]
        for turn in history:
            out += [USR] + self.encode_text(turn["user"])
            out += [AST] + self.encode_text(turn["assistant"])
        out += [USR] + self.encode_text(user_text) + [AST]

        if max_tokens and len(out) > max_tokens:
            out = [BOS] + out[-(max_tokens - 1):]
        return out
