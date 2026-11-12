"""Salamandra tokenizer fertility na bosanskom (brza CPU provjera, Faza 0).

Koristenje: python scripts/fertility_bs.py
"""
from transformers import AutoTokenizer

UZORCI = [
    "Bosna i Hercegovina je država u jugoistočnoj Evropi.",
    "Jelah je naseljeno mjesto u opštini Tešanj.",
    "Vlada je donijela odluku o izmjenama i dopunama zakona o porezu.",
    "Dijete je pročitalo knjigu o životinjama iz šume.",
    "Hoće li sutra biti lijepo vrijeme za izlet na planinu?",
    "Ugovor stupa na snagu danom potpisivanja obje strane.",
    "Molimo vas da popunite obrazac velikim štampanim slovima.",
    "Pritisnite dugme da potvrdite brisanje datoteke s računara.",
]


def main():
    t = AutoTokenizer.from_pretrained("BSC-LT/salamandra-2b", trust_remote_code=False)
    tt = sum(len(t.encode(u)) for u in UZORCI)
    rr = sum(len(u.split()) for u in UZORCI)
    print(f"vocab: {t.vocab_size}, prosjek: {tt / rr:.2f} tokena po rijeci")


if __name__ == "__main__":
    main()
