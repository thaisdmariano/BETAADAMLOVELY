#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import re as _re
from typing import List, Dict, Tuple, Set, Any, Optional
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import streamlit as st

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

import sys
import time

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE ARQUIVOS E CONSTANTES
# ────────────────────────────────────────────────────────────────────────────────

ARQUIVO_MEMORIA = "Adam_Lovely_memory.json"
ARQUIVO_INCONSCIENTE = "Adam_Lovely_inconscious.json"
EMBED_DIM = 16
HIDDEN_DIM = 64
PATIENCE = 5
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
UNK = "<UNK>"
UNK_VAL = -1.0
N_GRAM = 2  # Tamanho do n-grama (2 para bigrams)

SENHA_ADMIN = "adam123"  # Senha para acessar Gerenciar IMs e dados completos de teste


## INSEPA_TOKENIZER
def generate_ngrams(token: str, n: int) -> List[str]:
    """Gera n-gramas de caracteres de um token."""
    if len(token) < n:
        return [token]  # Se menor que n, retorna o token inteiro
    return [token[i:i + n] for i in range(len(token) - n + 1)]


def ckpt_path(dominio: str) -> str:
    return f"insepa_{dominio}.pt"


def Token(text: str) -> List[str]:
    """INSEPA tokenização: mantém palavras, pontuação, emojis, stopwords."""
    return _re.findall(r'\w+|[^\w\s]', text, _re.UNICODE)


def next_marker(prev: str) -> str:
    """Incrementa sem arredondar: 0.99 → 0.100"""
    mom, _, suf = prev.partition('.')
    if not mom.isdigit():
        raise ValueError(f"Marcador inválido: {prev!r}")
    return f"{mom}.{int(suf or '0') + 1}"


def generate_markers(start: str, count: int) -> List[str]:
    seq, cur = [], start
    for _ in range(count):
        cur = next_marker(cur)
        seq.append(cur)
    return seq


## INSEPA_UTILS
def carregar_json(caminho: str, default: dict) -> dict:
    if not os.path.exists(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default
    with open(caminho, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Sempre atualizar session_state
    if caminho == ARQUIVO_MEMORIA:
        st.session_state.memoria = data
    elif caminho == ARQUIVO_INCONSCIENTE:
        st.session_state.inconsciente = data
    return data


def salvar_json(caminho: str, data: dict) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if caminho == ARQUIVO_MEMORIA:
        st.session_state.memoria = data
    elif caminho == ARQUIVO_INCONSCIENTE:
        st.session_state.inconsciente = data


def garantir_pontuacao(txt: str) -> str:
    txt = txt.strip()
    return txt if txt and txt[-1] in ".!?" else (txt + "." if txt else "")


def normalize_collapse_spaces(txt: str) -> str:
    return _re.sub(r'\s+', ' ', txt).strip()


def normalize_separators(txt: str) -> str:
    return _re.sub(r'\s*([,.;:])\s*', '', txt).strip()


def normalize(txt: str) -> str:
    for fn in (normalize_collapse_spaces, normalize_separators):
        txt = fn(txt)
    return txt.lower()


def variar_texto(texto: str, bloco: dict, dominio: str) -> str:
    """Varia o texto substituindo tokens por suas variações aleatórias baseadas nas vars do inconsciente."""
    tokens = Token(texto)
    inconsciente = st.session_state.inconsciente
    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return texto
    variado = []
    for tok in tokens:
        marcador = None
        for m, data in bloco_inco["SAÍDA"].items():
            if data["token"] == tok:
                marcador = m
                break
        if marcador:
            # Filtrar vars válidas (remover "0.0" que é placeholder)
            valid_vars = [v for v in data["vars"] if v != "0.0"]
            if valid_vars and len(valid_vars) > 0:
                chosen_var = random.choice(valid_vars + [tok])
            else:
                chosen_var = tok
        else:
            chosen_var = tok
        variado.append(chosen_var)
    return ' '.join(variado)


def get_variations_for_tokens(im_id: str, bloco_id: int, campo: str, markers: List[str]) -> List[str]:
    """Obtém variações de tokens para marcadores específicos."""
    inconsciente = st.session_state.inconsciente
    bloco_inco = next((b for b in inconsciente["INCO"][im_id]["Blocos"] if b["Bloco_id"] == str(bloco_id)), None)
    if bloco_inco:
        variations = set()
        for marker in markers:
            if marker in bloco_inco[campo]:
                data = bloco_inco[campo][marker]
                variations.add(normalize(data["token"]))
                for var in data.get("vars", []):
                    variations.add(normalize(var))
        return list(variations)
    return []


def parse_text_reaction(prompt: str, reactions: Set[str]) -> Tuple[str, str]:
    s = prompt.strip()
    sorted_reactions = sorted(reactions, key=len, reverse=True)
    for reac in sorted_reactions:
        if reac and s.endswith(reac):
            txt = s[:-len(reac)].rstrip()
            return txt, reac
    return s, ""


def build_field_vocabs(memoria: dict, dominio: str) -> Dict[str, Dict[str, int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets = {"E": set(), "RE": set(), "CE": set(), "PIDE": set()}
    for b in blocos:
        t = b["entrada"]["tokens"]
        for f in sets:
            for tok in t.get(f, []):
                ngrams = generate_ngrams(tok, N_GRAM)
                sets[f].update(ngrams)
    return {
        f: {ng: i + 1 for i, ng in enumerate(sorted(sets[f]))}
        for f in sets
    }


def build_label_vocabs(memoria: dict, dominio: str) -> Dict[str, Dict[str, int]]:
    blocos = memoria["IM"][dominio]["blocos"]
    sets = {"texto": set(), "emoji": set(), "ctx": set()}
    for b in blocos:
        for s in b.get("saidas", []):
            for v in s.get("textos", []):
                sets["texto"].add(normalize(v))
            emo = s.get("reacao", "")
            if emo:   sets["emoji"].add(emo)
            ctx = s.get("contexto", "")
            if ctx:   sets["ctx"].add(normalize(ctx))
    return {
        f: {tok: i for i, tok in enumerate(sorted(sets[f]))}
        for f in sets
    }


## INSEPA_DATASET
class InsepaFieldDataset(Dataset):
    def __init__(self, memoria: dict, dominio: str):
        blocos = memoria["IM"][dominio]["blocos"]
        inconsciente = st.session_state.inconsciente
        self.ultimo_child_per_block = {}
        if dominio in inconsciente.get("INCO", {}):
            blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
            for bloco in blocos_inco:
                bloco_num = int(bloco["Bloco_id"])
                saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
                if saida_vals:
                    self.ultimo_child_per_block[bloco_num] = max(saida_vals)
                else:
                    self.ultimo_child_per_block[bloco_num] = 0.50

        # Coletar tokens únicos por campo diretamente do JSON
        sets = {"E": set(), "RE": set(), "CE": set(), "PIDE": set()}
        for b in blocos:
            for f in sets:
                sets[f] |= set(b["entrada"]["tokens"].get(f, []))

        self.v_E = {tok: i + 1 for i, tok in enumerate(sorted(sets["E"]))}
        self.v_RE = {tok: i + 1 for i, tok in enumerate(sorted(sets["RE"]))}
        self.v_CE = {tok: i + 1 for i, tok in enumerate(sorted(sets["CE"]))}
        self.v_PIDE = {tok: i + 1 for i, tok in enumerate(sorted(sets["PIDE"]))}
        self.v_E[UNK] = len(self.v_E)
        self.v_RE[UNK] = len(self.v_RE)
        self.v_CE[UNK] = len(self.v_CE)
        self.v_PIDE[UNK] = len(self.v_PIDE)

        # val_to_idx por campo: tokens únicos como chaves
        self.val_to_idx_E = {tok: i for i, tok in enumerate(sorted(sets["E"]))}
        self.val_to_idx_RE = {tok: i for i, tok in enumerate(sorted(sets["RE"]))}
        self.val_to_idx_CE = {tok: i for i, tok in enumerate(sorted(sets["CE"]))}
        self.val_to_idx_PIDE = {tok: i for i, tok in enumerate(sorted(sets["PIDE"]))}

        self.max_E = max(len(b["entrada"]["tokens"].get("E", [])) for b in blocos)
        self.max_RE = max(len(b["entrada"]["tokens"].get("RE", [])) for b in blocos)
        self.max_CE = max(len(b["entrada"]["tokens"].get("CE", [])) for b in blocos)
        self.max_PIDE = max(len(b["entrada"]["tokens"].get("PIDE", [])) for b in blocos)
        self.max_pos = max(self.max_E, self.max_RE, self.max_CE, self.max_PIDE)

        # Calcular max n-gramas por token
        all_tokens = set()
        for b in blocos:
            for field in ["E", "RE", "CE", "PIDE"]:
                all_tokens |= set(b["entrada"]["tokens"].get(field, []))
        self.max_ng = max(len(generate_ngrams(t, N_GRAM)) for t in all_tokens if t) if all_tokens else 1
        self.max_E_ng = self.max_E * self.max_ng
        self.max_RE_ng = self.max_RE * self.max_ng
        self.max_CE_ng = self.max_CE * self.max_ng
        self.max_PIDE_ng = self.max_PIDE * self.max_ng

        # calcula mom_size = maior mãe + 1
        max_mom = 0
        for b in blocos:
            for tok in b["entrada"]["tokens"].get("TOTAL", []):
                m = int(tok.split(".", 1)[0])
                if m > max_mom: max_mom = m
        self.mom_size = max_mom + 1

        # valores únicos para posições fixas (não usado agora, mas manter compatibilidade)
        vals = {float(t) for t in all_tokens if t}
        sorted_vals = sorted(vals)
        self.val_to_idx = {v: i + 1 for i, v in enumerate(sorted_vals)}  # índices de 1 em diante, 0 para padding
        self.num_vals = len(sorted_vals)

        # vocabulários de rótulos por bloco
        self.n_txt = max(len(b["saidas"][0]["textos"]) for b in blocos)
        self.n_emo = max(1, len(set(b["saidas"][0].get("reacao", "") for b in blocos if b["saidas"][0].get("reacao"))))
        self.n_ctx = max(1, len(set(
            normalize(b["saidas"][0].get("contexto", "")) for b in blocos if b["saidas"][0].get("contexto"))))

        self.pares: List[Tuple[Dict, Dict]] = []
        for b in blocos:
            bloco_id = b["bloco_id"]
            max_val = self.ultimo_child_per_block.get(bloco_id, 0.50)

            # Usar tokens fixos do JSON
            E_tokens = b["entrada"]["tokens"].get("E", [])
            RE_tokens = b["entrada"]["tokens"].get("RE", [])
            CE_tokens = b["entrada"]["tokens"].get("CE", [])
            PIDE_tokens = b["entrada"]["tokens"].get("PIDE", [])

            # Gerar n-gramas e ids
            E_ngrams = [generate_ngrams(t, N_GRAM) for t in E_tokens]
            RE_ngrams = [generate_ngrams(t, N_GRAM) for t in RE_tokens]
            CE_ngrams = [generate_ngrams(t, N_GRAM) for t in CE_tokens]
            PIDE_ngrams = [generate_ngrams(t, N_GRAM) for t in PIDE_tokens]

            E_ids = [self.v_E.get(ng, self.v_E.get(UNK, 0)) for nglist in E_ngrams for ng in nglist]
            RE_ids = [self.v_RE.get(ng, self.v_RE.get(UNK, 0)) for nglist in RE_ngrams for ng in nglist]
            CE_ids = [self.v_CE.get(ng, self.v_CE.get(UNK, 0)) for nglist in CE_ngrams for ng in nglist]
            PIDE_ids = [self.v_PIDE.get(ng, self.v_PIDE.get(UNK, 0)) for nglist in PIDE_ngrams for ng in nglist]

            E_ids += [0] * (self.max_E_ng - len(E_ids))
            RE_ids += [0] * (self.max_RE_ng - len(RE_ids))
            CE_ids += [0] * (self.max_CE_ng - len(CE_ids))
            PIDE_ids += [0] * (self.max_PIDE_ng - len(PIDE_ids))

            # índices de valores para embedding (mantém tokens)
            E_val_idxs = [self.val_to_idx_E.get(t, 0) for t in E_tokens]
            RE_val_idxs = [self.val_to_idx_RE.get(t, 0) for t in RE_tokens]
            CE_val_idxs = [self.val_to_idx_CE.get(t, 0) for t in CE_tokens]
            PIDE_val_idxs = [self.val_to_idx_PIDE.get(t, 0) for t in PIDE_tokens]
            E_val_idxs += [0] * (self.max_E - len(E_val_idxs))
            RE_val_idxs += [0] * (self.max_RE - len(RE_val_idxs))
            CE_val_idxs += [0] * (self.max_CE - len(CE_val_idxs))
            PIDE_val_idxs += [0] * (self.max_PIDE - len(PIDE_val_idxs))

            # função para gerar valores, mães e posições
            def build_feats(tokens, maxlen):
                vals = [float(tok) for tok in tokens]
                moms = [int(tok.split(".", 1)[0]) for tok in tokens]
                if vals:
                    min_v, max_v = min(vals), max(vals)
                    pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
                else:
                    pos = []
                pad = maxlen - len(tokens)
                vals += [0.0] * pad
                moms += [0] * pad
                pos += [0.0] * pad
                return vals, moms, pos

            E_vals, E_moms, E_pos = build_feats(E_tokens, self.max_E)
            RE_vals, RE_moms, RE_pos = build_feats(RE_tokens, self.max_RE)
            CE_vals, CE_moms, CE_pos = build_feats(CE_tokens, self.max_CE)
            PI_vals, PI_moms, PI_pos = build_feats(PIDE_tokens, self.max_PIDE)

            for texto_idx, texto in enumerate(b["saidas"][0]["textos"]):
                # calcula pos_label = média dos valores dos tokens no bloco
                all_vals = []
                for tokens in [E_tokens, RE_tokens, CE_tokens, PIDE_tokens]:
                    all_vals.extend([float(t) for t in tokens])
                pos_label = sum(all_vals) / len(all_vals) if all_vals else 0.0

                y = {
                    "texto": texto_idx,
                    "emoji": 0,  # sempre 0, pois reacao é a mesma para todos os textos
                    "ctx": 0,    # sempre 0, pois contexto é o mesmo para todos os textos
                    "pos": pos_label,
                }
                x = {
                    "E": E_ids, "E_val": E_vals, "E_mom": E_moms, "E_pos": E_pos, "E_val_idx": E_val_idxs,
                    "RE": RE_ids, "RE_val": RE_vals, "RE_mom": RE_moms, "RE_pos": RE_pos, "RE_val_idx": RE_val_idxs,
                    "CE": CE_ids, "CE_val": CE_vals, "CE_mom": CE_moms, "CE_pos": CE_pos, "CE_val_idx": CE_val_idxs,
                    "PIDE": PIDE_ids, "PIDE_val": PI_vals, "PIDE_mom": PI_moms, "PIDE_pos": PI_pos,
                    "PIDE_val_idx": PIDE_val_idxs,
                }
                self.pares.append((x, y))

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int):
        x, y = self.pares[idx]
        x_t = {
            "E": torch.tensor(x["E"], dtype=torch.long),
            "E_val": torch.tensor(x["E_val"], dtype=torch.float32),
            "E_mom": torch.tensor(x["E_mom"], dtype=torch.long),
            "E_pos": torch.tensor(x["E_pos"], dtype=torch.long),
            "E_val_idx": torch.tensor(x["E_val_idx"], dtype=torch.long),

            "RE": torch.tensor(x["RE"], dtype=torch.long),
            "RE_val": torch.tensor(x["RE_val"], dtype=torch.float32),
            "RE_mom": torch.tensor(x["RE_mom"], dtype=torch.long),
            "RE_pos": torch.tensor(x["RE_pos"], dtype=torch.long),
            "RE_val_idx": torch.tensor(x["RE_val_idx"], dtype=torch.long),

            "CE": torch.tensor(x["CE"], dtype=torch.long),
            "CE_val": torch.tensor(x["CE_val"], dtype=torch.float32),
            "CE_mom": torch.tensor(x["CE_mom"], dtype=torch.long),
            "CE_pos": torch.tensor(x["CE_pos"], dtype=torch.long),
            "CE_val_idx": torch.tensor(x["CE_val_idx"], dtype=torch.long),

            "PIDE": torch.tensor(x["PIDE"], dtype=torch.long),
            "PIDE_val": torch.tensor(x["PIDE_val"], dtype=torch.float32),
            "PIDE_mom": torch.tensor(x["PIDE_mom"], dtype=torch.long),
            "PIDE_pos": torch.tensor(x["PIDE_pos"], dtype=torch.long),
            "PIDE_val_idx": torch.tensor(x["PIDE_val_idx"], dtype=torch.long),
        }
        y_t = {
            "texto": torch.tensor(y["texto"], dtype=torch.long),
            "emoji": torch.tensor(y["emoji"], dtype=torch.long),
            "ctx": torch.tensor(y["ctx"], dtype=torch.long),
            "pos": torch.tensor(y["pos"], dtype=torch.float32),
        }
        return x_t, y_t


## INSEPA_MODEL
class AdamSegmentado(nn.Module):
    def __init__(self,
                 nE: int, nRE: int, nCE: int, nPIDE: int,
                 mom_size: int,
                 num_vals_E: int, num_vals_RE: int, num_vals_CE: int, num_vals_PIDE: int,
                 n_txt: int, n_emo: int, n_ctx: int,
                 max_E: int, max_RE: int, max_CE: int, max_PIDE: int, max_ng: int):
        super().__init__()
        # Embeddings por valor, separados por campo (treináveis)
        self.em_Eval = nn.Embedding(num_vals_E, EMBED_DIM)
        self.em_REval = nn.Embedding(num_vals_RE, EMBED_DIM)
        self.em_CEval = nn.Embedding(num_vals_CE, EMBED_DIM)
        self.em_PIDEval = nn.Embedding(num_vals_PIDE, EMBED_DIM)

        # Embeddings para tokens, mães e projeções de posição
        self.em_E = nn.Embedding(nE, EMBED_DIM)
        self.em_RE = nn.Embedding(nRE, EMBED_DIM)
        self.em_CE = nn.Embedding(nCE, EMBED_DIM)
        self.em_PIDE = nn.Embedding(nPIDE, EMBED_DIM)
        self.em_Emom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_REmom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_CEmom = nn.Embedding(mom_size, EMBED_DIM)
        self.em_PIDEmom = nn.Embedding(mom_size, EMBED_DIM)
        self.proj_Epos = nn.Linear(1, EMBED_DIM)
        self.proj_REpos = nn.Linear(1, EMBED_DIM)
        self.proj_CEpos = nn.Linear(1, EMBED_DIM)
        self.proj_PIDEpos = nn.Linear(1, EMBED_DIM)

        self.max_E = max_E
        self.max_RE = max_RE
        self.max_CE = max_CE
        self.max_PIDE = max_PIDE
        self.max_ng = max_ng

        # Transformer Encoder para processar sequência de campos
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=HIDDEN_DIM),
            num_layers=1
        )

        self.fc1 = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.act = nn.ReLU()

        # Cabeças de saída
        self.h_txt = nn.Linear(HIDDEN_DIM, n_txt)
        self.h_emo = nn.Linear(HIDDEN_DIM, n_emo)
        self.h_ctx = nn.Linear(HIDDEN_DIM, n_ctx)
        self.h_pos = nn.Linear(HIDDEN_DIM, 1)  # Nova cabeça para posição (regressão)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = x["E"].shape[0]
        # Campo E
        eE_tok = self.em_E(x["E"]).view(batch, self.max_E, self.max_ng, EMBED_DIM).mean(dim=2)
        eE_val = self.em_Eval(x["E_val_idx"])
        eE_mom = self.em_Emom(x["E_mom"])
        eE_pos = self.proj_Epos(x["E_val"].unsqueeze(-1))
        eE = (eE_tok + eE_val + eE_mom + eE_pos).mean(dim=1)
        # Campo RE
        eRE_tok = self.em_RE(x["RE"]).view(batch, self.max_RE, self.max_ng, EMBED_DIM).mean(dim=2)
        eRE_val = self.em_REval(x["RE_val_idx"])
        eRE_mom = self.em_REmom(x["RE_mom"])
        eRE_pos = self.proj_REpos(x["RE_val"].unsqueeze(-1))
        eRE = (eRE_tok + eRE_val + eRE_mom + eRE_pos).mean(dim=1)
        # Campo CE
        eCE_tok = self.em_CE(x["CE"]).view(batch, self.max_CE, self.max_ng, EMBED_DIM).mean(dim=2)
        eCE_val = self.em_CEval(x["CE_val_idx"])
        eCE_mom = self.em_CEmom(x["CE_mom"])
        eCE_pos = self.proj_CEpos(x["CE_val"].unsqueeze(-1))
        eCE = (eCE_tok + eCE_val + eCE_mom + eCE_pos).mean(dim=1)
        # Campo PIDE
        ePI_tok = self.em_PIDE(x["PIDE"]).view(batch, self.max_PIDE, self.max_ng, EMBED_DIM).mean(dim=2)
        ePI_val = self.em_PIDEval(x["PIDE_val_idx"])
        ePI_mom = self.em_PIDEmom(x["PIDE_mom"])
        ePI_pos = self.proj_PIDEpos(x["PIDE_val"].unsqueeze(-1))
        ePIDE = (ePI_tok + ePI_val + ePI_mom + ePI_pos).mean(dim=1)

        # Agrega e classifica
        # Empilhar embeddings dos campos em sequência
        seq = torch.stack([eE, eRE, eCE, ePIDE], dim=1)  # (batch, 4, EMBED_DIM)
        seq = seq.permute(1, 0, 2)  # (4, batch, EMBED_DIM)
        transformed = self.transformer(seq)  # (4, batch, EMBED_DIM)
        transformed = transformed.permute(1, 0, 2)  # (batch, 4, EMBED_DIM)
        h = transformed.mean(dim=1)  # (batch, EMBED_DIM)
        h = self.act(self.fc1(h))
        return {
            "texto": self.h_txt(h),
            "emoji": self.h_emo(h),
            "ctx": self.h_ctx(h),
            "pos": self.h_pos(h),  # Posição como valor numérico
        }


## INSEPA_TRAIN
def train(memoria: dict, dominio: str) -> None:
    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ds = InsepaFieldDataset(memoria, dominio)
    n = len(ds)
    ckpt = ckpt_path(dominio)

    idxs = list(range(n))
    random.shuffle(idxs)
    vsz = min(max(1, int(0.2 * n)), n - 1)  # garantir pelo menos 1 para treino
    vidx, tidx = idxs[:vsz], idxs[vsz:]
    if not tidx:  # se tidx vazio, usar todos para treino, sem val
        tidx = idxs
        vidx = []
    train_ld = DataLoader(Subset(ds, tidx), batch_size=min(BATCH_SIZE, len(tidx)), shuffle=True)
    val_ld = DataLoader(Subset(ds, vidx), batch_size=min(BATCH_SIZE, len(vidx))) if vidx else None

    model = AdamSegmentado(
        nE=len(ds.v_E), nRE=len(ds.v_RE),
        nCE=len(ds.v_CE), nPIDE=len(ds.v_PIDE),
        mom_size=ds.mom_size,
        num_vals_E=len(ds.val_to_idx_E), num_vals_RE=len(ds.val_to_idx_RE),
        num_vals_CE=len(ds.val_to_idx_CE), num_vals_PIDE=len(ds.val_to_idx_PIDE),
        n_txt=ds.n_txt, n_emo=ds.n_emo,
        n_ctx=ds.n_ctx,
        max_E=ds.max_E, max_RE=ds.max_RE, max_CE=ds.max_CE, max_PIDE=ds.max_PIDE, max_ng=ds.max_ng
    )
    opt = optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    best, wait, prev_val = float("inf"), 0, None
    progress_bar = st.progress(0)
    status_text = st.empty()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_ld:
            opt.zero_grad()
            out = model(x)
            loss = (
                    ce(out["texto"], y["texto"]) +
                    ce(out["emoji"], y["emoji"]) +
                    ce(out["ctx"], y["ctx"]) +
                    mse(out["pos"], y["pos"])
            )
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        if val_ld:
            with torch.no_grad():
                for x, y in val_ld:
                    out = model(x)
                    val_loss += (
                            ce(out["texto"], y["texto"]).item() +
                            ce(out["emoji"], y["emoji"]).item() +
                            ce(out["ctx"], y["ctx"]).item() +
                            mse(out["pos"], y["pos"]).item()
                    )
            val_loss /= len(val_ld)
        else:
            val_loss = float("inf")  # sem validação, usar inf para não salvar

        if prev_val is None or val_loss < best:
            best, wait = val_loss, 0
            torch.save((
                model.state_dict(),
                ds.max_E, ds.max_RE, ds.max_CE, ds.max_PIDE,
                ds.mom_size, ds.val_to_idx_E, ds.val_to_idx_RE, ds.val_to_idx_CE, ds.val_to_idx_PIDE,
                ds.v_E, ds.v_RE, ds.v_CE, ds.v_PIDE,
                ds.n_txt, ds.n_emo, ds.n_ctx,
                ds.max_ng
            ), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        prev_val = val_loss
        progress_bar.progress(ep / EPOCHS)
        status_text.text(f"Época {ep}/{EPOCHS}, Val Loss: {val_loss:.4f}")

    st.success(f"✅ Treino concluído. best_val_loss={best:.4f}")
    
    # Salvar backup do JSON usado para treinamento
    backup_memoria = f"Adam_Lovely_memory_backup_{dominio}_{int(time.time())}.json"
    salvar_json(backup_memoria, memoria)
    st.info(f"📁 Backup do JSON salvo como: {backup_memoria}")


def generate_insight(bloco, chosen=None):
    principles = "Texto vem com emoção (E+RE). Reação (RE) simboliza emoção. Contexto é noção breve. Reflexão dá significado a Texto/Reação/Contexto. Criação baseia-se em similares (contextos, reações, textos). União dos 8 campos = conhecimento concreto (razão + sentimento). Ausência = abstrato (opinião/incompleto)."
    if bloco["entrada"]["texto"] and bloco["entrada"].get("reacao") and bloco["saidas"][0].get("contexto"):
        ep_txt = bloco["entrada"]["texto"]
        ep_reac = bloco["entrada"]["reacao"]
        contexto = bloco["saidas"][0]["contexto"]
        if chosen is None:
            chosen = bloco["saidas"][0]["textos"][0]
        emoji = bloco["saidas"][0].get("reacao", "")
        return f"{principles} Baseado na entrada '{ep_txt}', reação '{ep_reac}' e contexto '{contexto}', conclui que '{chosen} {emoji}' é a resposta mais adequada."
    return principles


def infer(memoria: dict, dominio: str) -> None:
    """
    Interface de chat inovadora para inferência.
    """
    import os, torch, random
    # parse_text_reaction, normalize, ckpt_path, train, AdamSegmentado já disponíveis

    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint — treine primeiro.")
        train(memoria, dominio)
        return

    (state,
     maxE, maxRE, maxCE, maxPIDE,
     mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
     vE, vRE, vCE, vPIDE,
     n_txt, n_emo, n_ctx,
     max_ng
     ) = torch.load(ckpt)

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE),
        nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        n_txt=n_txt, n_emo=n_emo,
        n_ctx=n_ctx,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        st.warning(f"⚠️ Checkpoint incompatível devido a mudanças na arquitetura: {e}. Retreinando...")
        train(memoria, dominio)
        return
    model.eval()

    model_params = (state, maxE, maxRE, maxCE, maxPIDE, mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE, vE, vRE, vCE, vPIDE, n_txt, n_emo, n_ctx, max_ng)

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = st.session_state.inconsciente
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
        for bloco in blocos_inco:
            bloco_num = int(bloco["Bloco_id"])
            saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
            if saida_vals:
                ultimo_child_per_block[bloco_num] = max(saida_vals)
            else:
                ultimo_child_per_block[bloco_num] = 0.50

    # Coletar todas as reações possíveis, incluindo variações
    all_possible_reactions = set()
    inconsciente = st.session_state.inconsciente
    for b in blocos:
        reac = b["entrada"].get("reacao", "")
        if reac:
            all_possible_reactions.add(reac)
        # Obter vars originais para RE
        bloco_inco = next((bi for bi in inconsciente["INCO"][dominio]["Blocos"] if bi["Bloco_id"] == str(b["bloco_id"])), None)
        if bloco_inco:
            for marker in b["entrada"]["tokens"].get("RE", []):
                if marker in bloco_inco["Entrada"]:
                    data = bloco_inco["Entrada"][marker]
                    all_possible_reactions.add(data["token"])
                    for var in data.get("vars", []):
                        if var != "0.0":
                            all_possible_reactions.add(var)

    # Mostrar nome do IM
    nome_im = memoria["IM"][dominio].get("nome", f"IM_{dominio}")
    genero = memoria["IM"][dominio].get("genero", "feminino")
    voz = memoria["IM"][dominio].get("voz", None)
    st.write(f"**Conversando com: {nome_im}**")

    # Inicializar histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "variation" not in st.session_state:
        st.session_state.variation = 0
    if "current_bloco" not in st.session_state:
        st.session_state.current_bloco = None
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "conversa_blocos" not in st.session_state:
        st.session_state.conversa_blocos = []

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Mostrar áudio se existir
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3')

    def featurize(field: str, bloco: dict, max_len: int, vocab: dict, val_to_idx: dict, max_ng: int):
        tokens = bloco["entrada"]["tokens"].get(field, [])
        ngrams_list = [generate_ngrams(t, N_GRAM) for t in tokens]
        ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
        val_idxs = [val_to_idx.get(t, 0) for t in tokens]
        vals = [float(t) for t in tokens]
        moms = [int(t.split(".", 1)[0]) for t in tokens]
        if vals:
            min_v, max_v = min(vals), max(vals)
            pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
        else:
            pos = []
        pad_ids = (max_len * max_ng) - len(ids)
        pad_vals = max_len - len(tokens)
        ids += [0] * pad_ids
        val_idxs += [0] * pad_vals
        vals += [0.0] * pad_vals
        moms += [0] * pad_vals
        pos += [0.0] * pad_vals
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([val_idxs], dtype=torch.long),
            torch.tensor([vals], dtype=torch.float32),
            torch.tensor([moms], dtype=torch.long),
            torch.tensor([pos], dtype=torch.float32),
        )

    # Entrada do usuário
    if prompt := st.chat_input("Digite sua mensagem + reação (ex: Olá 😊)"):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        cmd = prompt.lower().strip()

        if cmd == "sair":
            st.session_state.messages.append({"role": "assistant", "content": "👋 Até mais!"})
            with st.chat_message("assistant"):
                st.markdown("👋 Até mais!")
            st.session_state.messages = []
            st.session_state.variation = 0
            st.session_state.current_bloco = None
            st.session_state.conversa_blocos = []
            return

        if cmd == "reiniciar":
            st.session_state.messages.append({"role": "assistant", "content": "🔄 Conversa reiniciada. Histórico limpo."})
            with st.chat_message("assistant"):
                st.markdown("🔄 Conversa reiniciada. Histórico limpo.")
            st.session_state.messages = []
            st.session_state.variation = 0
            st.session_state.current_bloco = None
            st.session_state.conversa_blocos = []
            return

        if cmd == "insight" and st.session_state.current_bloco:
            bloco = st.session_state.current_bloco
            ep_txt = bloco["entrada"]["texto"]
            ep_reac = bloco["entrada"].get("reacao", "")
            contexto = bloco["saidas"][0].get("contexto", "")
            emoji = bloco["saidas"][0].get("reacao", "")
            texts = bloco["saidas"][0]["textos"]
            chosen = texts[st.session_state.variation]
            insight_msg = f"💡 De acordo com a expressão “{ep_txt}”, a reação “{ep_reac}” e o contexto “{contexto}”, conclui que “{chosen} {emoji}” é a resposta mais adequada."
            st.session_state.messages.append({"role": "assistant", "content": insight_msg})
            with st.chat_message("assistant"):
                st.markdown(insight_msg)
            # Armazenar a última resposta para like
            st.session_state.last_response = insight_msg
            st.session_state.last_bloco_id = str(bloco["bloco_id"])
            st.rerun()

        # Parse entrada normal
        txt, reac = parse_text_reaction(prompt, all_possible_reactions)
        bloco = None
        for b in blocos:
            # Verificar Multivars_Entrada primeiro (frases completas)
            if txt in b["entrada"].get("Multivars_Entrada", []):
                bloco = b
                break
            # Matching por tokens e vars
            txt_variations = get_variations_for_tokens(dominio, b["bloco_id"], "Entrada", b["entrada"]["tokens"]["E"])
            reac_variations = get_variations_for_tokens(dominio, b["bloco_id"], "Entrada", b["entrada"]["tokens"].get("RE", []))
            txt_tokens = Token(txt)
            if all(normalize(t) in txt_variations for t in txt_tokens) and (not b["entrada"]["tokens"].get("RE") or (reac and normalize(reac) in reac_variations)):
                bloco = b
                break
        if bloco is None:
            # Verificar se o texto matching mas a reação não
            for b in blocos:
                txt_variations = get_variations_for_tokens(dominio, b["bloco_id"], "Entrada", b["entrada"]["tokens"]["E"])
                if all(normalize(t) in txt_variations for t in txt_tokens):
                    # Texto matching, mas reação não
                    st.session_state.messages.append({"role": "assistant", "content": "Hmm parece que falta emoção em sua expressão. Por favor verifique seu emoji."})
                    with st.chat_message("assistant"):
                        st.markdown("Hmm parece que falta emoção em sua expressão. Por favor verifique seu emoji.")
                    st.rerun()
            # Se não encontrou nem texto, tentar histórico ou erro
            if st.session_state.conversa_blocos:
                bloco = st.session_state.conversa_blocos[-1]
                st.session_state.messages.append({"role": "assistant", "content": f"💭 Continuando do bloco {bloco['bloco_id']}..."})
                with st.chat_message("assistant"):
                    st.markdown(f"💭 Continuando do bloco {bloco['bloco_id']}...")
            else:
                error_msg = "Desculpe mas seu texto e emoji não existem neste universo. Por favor verifique sua mensagem e tente novamente."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.last_valid = False
                st.rerun()

        # Adicionar bloco ao histórico se for novo
        if bloco and bloco not in st.session_state.conversa_blocos:
            st.session_state.conversa_blocos.append(bloco)

        st.session_state.current_bloco = bloco
        st.session_state.variation = 0
        st.session_state.last_valid = True

        # Pensamento Divino: Validação dos 8 campos
        bloco = st.session_state.current_bloco
        fields_check = {
            "E": len(bloco["entrada"]["tokens"].get("E", [])) > 0,
            "RE": len(bloco["entrada"]["tokens"].get("RE", [])) > 0,
            "CE": len(bloco["entrada"]["tokens"].get("CE", [])) > 0,
            "PIDE": len(bloco["entrada"]["tokens"].get("PIDE", [])) > 0,
            "S": len(bloco["saidas"][0]["tokens"].get("S", [])) > 0,
            "RS": len(bloco["saidas"][0]["tokens"].get("RS", [])) > 0,
            "CS": len(bloco["saidas"][0]["tokens"].get("CS", [])) > 0
        }
        missing_fields = [k for k, v in fields_check.items() if not v]
        if missing_fields:
            st.write(f"Pensamento Divino: Campos faltantes para conhecimento concreto: {', '.join(missing_fields)}. Continuando...")
        else:
            st.write("Pensamento Divino Ativado: Todos os 8 campos divinos estão presentes no conhecimento prévio dos blocos (E, RE, CE, PIDE, S, RS, CS). Conhecimento concreto alcançado.")

        # Gerar Insight se condições atendidas
        insight = generate_insight(st.session_state.current_bloco)
        if insight:
            st.write(insight)

        # Preparo e forward
        max_val = ultimo_child_per_block.get(bloco["bloco_id"], 0.50)
        E_ids, E_val_idxs, E_val, E_mom, E_pos = featurize("E", bloco, maxE, vE, val_to_idx_E, max_ng)
        RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos = featurize("RE", bloco, maxRE, vRE, val_to_idx_RE, max_ng)
        CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE", bloco, maxCE, vCE, val_to_idx_CE, max_ng)
        PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", bloco, maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

        x = {
            "E": E_ids, "E_val": E_val, "E_mom": E_mom, "E_pos": E_pos, "E_val_idx": E_val_idxs,
            "RE": RE_ids, "RE_val": RE_val, "RE_mom": RE_mom, "RE_pos": RE_pos, "RE_val_idx": RE_val_idxs,
            "CE": CE_ids, "CE_val": CE_val, "CE_mom": CE_mom, "CE_pos": CE_pos, "CE_val_idx": CE_val_idxs,
            "PIDE": PI_ids, "PIDE_val": PI_val, "PIDE_mom": PI_mom, "PIDE_pos": PI_pos, "PIDE_val_idx": PI_val_idxs,
        }

        with torch.no_grad():
            out = model(x)

        texts = bloco["saidas"][0]["textos"]
        emoji = bloco["saidas"][0].get("reacao", "")
        # Gerar variações para cada texto, sem misturar textos
        all_variations = []
        for txt in texts:
            variations = [txt]  # Sempre incluir o original
            # Adicionar variações do inconsciente se existirem
            bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
            if bloco_inco:
                for marker, data in bloco_inco["SAÍDA"].items():
                    if data["token"] in txt:
                        # Adicionar vars válidas (remover "0.0" que é placeholder)
                        valid_vars = [v for v in data["vars"] if v != "0.0"]
                        for var in valid_vars:
                            # Substituir o token no texto pela var
                            varied_txt = txt.replace(data["token"], var)
                            variations.append(varied_txt)
            all_variations.extend(variations)  # Adicionar variações deste texto
        
        # Adicionar Multivars_Saída
        multivars_saida = bloco["saidas"][0].get("Multivars_Saída", [])
        all_variations.extend(multivars_saida)
        
        # Escolher uma variação aleatoriamente com pesos
        bloco_id = str(bloco["bloco_id"])
        chosen = weighted_choice(all_variations, bloco_id)
        
        chosen = variar_texto(chosen, bloco, dominio)
        
        response = f"{chosen} {emoji}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        # Armazenar a última resposta para like
        st.session_state.last_response = chosen
        st.session_state.last_bloco_id = str(bloco["bloco_id"])
        # Armazenar a última resposta para like
        st.session_state.last_response = chosen
        st.session_state.last_bloco_id = str(bloco["bloco_id"])
        # Generate speech - sistema otimizado: Edge TTS para vozes premium, gTTS para leves, pyttsx3 para outras
        if TTS_AVAILABLE:
            try:
                if voz and voz.startswith('edge-') and EDGE_TTS_AVAILABLE:
                    # Usar Edge TTS para vozes premium do Microsoft Edge
                    voice_name = voz.split('-', 1)[1]
                    
                    # Mapeamento de códigos de voz simplificados para vozes Edge TTS
                    
                    # Vozes Femininas
                    edge_voice_map_female = {
                        'pt-br': 'pt-BR-FranciscaNeural',  # Feminina
                        'pt-pt': 'pt-PT-RaquelNeural',     # Feminina
                        'en': 'en-US-AriaNeural',          # Feminina
                        'en-us': 'en-US-AriaNeural',       # Feminina
                        'en-gb': 'en-GB-SoniaNeural',      # Feminina
                        'es': 'es-ES-ElviraNeural',        # Feminina
                        'es-us': 'es-US-PalomaNeural',     # Feminina
                        'fr': 'fr-FR-DeniseNeural',        # Feminina
                        'de': 'de-DE-KatjaNeural',         # Feminina
                        'it': 'it-IT-ElsaNeural',          # Feminina
                        'ja': 'ja-JP-NanamiNeural',        # Feminina
                        'ko': 'ko-KR-SunHiNeural',         # Feminina
                        'ru': 'ru-RU-SvetlanaNeural',      # Feminina
                        'ar': 'ar-SA-ZariyahNeural',       # Feminina
                        'hi': 'hi-IN-SwaraNeural',         # Feminina
                        'female': 'en-US-AriaNeural',      # Feminina
                    }
                    
                    # Vozes Masculinas
                    edge_voice_map_male = {
                        'pt-br-male': 'pt-BR-AntonioNeural',    # Masculina
                        'en-male': 'en-US-AndrewNeural',        # Masculina
                        'es-male': 'es-ES-AlvaroNeural',        # Masculina
                        'fr-male': 'fr-FR-HenriNeural',         # Masculina
                        'de-male': 'de-DE-ConradNeural',        # Masculina
                        'it-male': 'it-IT-DiegoNeural',         # Masculina
                        'ja-male': 'ja-JP-KeitaNeural',         # Masculina
                        'ko-male': 'ko-KR-InJoonNeural',        # Masculina
                        'ru-male': 'ru-RU-DmitryNeural',        # Masculina
                        'ar-male': 'ar-SA-HamedNeural',         # Masculina
                        'hi-male': 'hi-IN-MadhurNeural',        # Masculina
                        'male': 'en-US-ZiraNeural',             # Masculina (nota: Zira é feminino, mas usado como padrão masculino)
                    }
                    
                    # Combinar dicionários
                    edge_voice_map = {**edge_voice_map_female, **edge_voice_map_male}
                    
                    selected_voice = edge_voice_map.get(voice_name, 'en-US-AriaNeural')
                    
                    import asyncio
                    import io
                    
                    async def generate_edge_audio():
                        communicate = edge_tts.Communicate(chosen, selected_voice)
                        audio_data = b""
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                audio_data += chunk["data"]
                        return audio_data
                    
                    # Executar de forma síncrona
                    audio_bytes = asyncio.run(generate_edge_audio())
                    
                    if audio_bytes and len(audio_bytes) > 0:
                        # Armazenar em session_state e reproduzir diretamente
                        st.session_state.last_audio = audio_bytes
                        st.audio(st.session_state.last_audio, format='audio/mp3')
                        st.success(f"🎵 Áudio gerado com Edge TTS '{selected_voice}': {len(audio_bytes)} bytes")
                    else:
                        st.error("❌ Falha ao gerar arquivo de áudio com Edge TTS.")
                elif voz and voz.startswith('gtts-') and GTTS_AVAILABLE:
                    lang_code = voz.split('-', 1)[1]
                    
                    # Mapear códigos de idioma do gTTS
                    lang_map = {
                        'pt-br': 'pt-br',
                        'pt-pt': 'pt-pt', 
                        'en': 'en',
                        'en-us': 'en',
                        'en-gb': 'en',
                        'es': 'es',
                        'es-us': 'es',
                        'fr': 'fr',
                        'de': 'de',
                        'it': 'it',
                        'ja': 'ja',
                        'ko': 'ko',
                        'ru': 'ru',
                        'ar': 'ar',
                        'hi': 'hi'
                    }
                    
                    if lang_code in lang_map:
                        from gtts import gTTS
                        import io
                        
                        # Gerar áudio com gTTS
                        tts = gTTS(text=chosen, lang=lang_map[lang_code], slow=False)
                        
                        # Salvar em buffer de memória
                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        audio_bytes = audio_buffer.read()
                        
                        if audio_bytes and len(audio_bytes) > 0:
                            # Armazenar em session_state e reproduzir diretamente
                            st.session_state.last_audio = audio_bytes
                            st.audio(st.session_state.last_audio, format='audio/mp3')
                            st.success(f"🎵 Áudio gerado com gTTS '{lang_code}': {len(audio_bytes)} bytes")
                        else:
                            st.error("❌ Falha ao gerar arquivo de áudio com gTTS.")
                    else:
                        st.warning(f"Idioma '{lang_code}' não suportado pelo gTTS.")
                else:
                    # Usar pyttsx3 para vozes automáticas ou quando gTTS não disponível
                    import pyttsx3
                    engine = pyttsx3.init()

                    # Configurar voz baseada no gênero do IM
                    voices = engine.getProperty('voices')
                    if voz:
                        # Se uma voz específica foi selecionada, tentar usar ela
                        selected_voice = next((v for v in voices if v.name == voz), voices[0] if voices else None)
                    else:
                        # Seleção automática baseada no gênero
                        if genero == "masculino":
                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                        elif genero == "feminino":
                            selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                        else:
                            selected_voice = random.choice(voices) if voices else None

                    if selected_voice:
                        engine.setProperty('voice', selected_voice.id)
                        engine.setProperty('rate', 180)  # Velocidade um pouco mais rápida
                        engine.setProperty('volume', 0.9)  # Volume alto

                        # Reproduzir diretamente sem salvar arquivo
                        engine.say(chosen)
                        engine.runAndWait()

                        st.success(f"🎵 Áudio reproduzido com sucesso! (Voz: {selected_voice.name})")
                    else:
                        st.warning("⚠️ Nenhuma voz do sistema encontrada. TTS pode não funcionar corretamente.")

            except Exception as e:
                import traceback
                st.error(f"Erro ao reproduzir áudio: {str(e)}")
                st.error("Detalhes do erro:")
                st.code(traceback.format_exc())
                st.warning("TTS falhou, mas a conversa continua normalmente.")

        # Pensamento Divino: Gerar saídas similares para aprovação
        similar_blocks = find_similar_blocks(st.session_state.current_bloco, memoria, dominio, model_params)
        if similar_blocks:
            st.write("Pensamento Divino: Saídas similares encontradas baseadas no conhecimento prévio dos blocos. Aprove uma para salvar como novo bloco:")
            for i, sim_bloco in enumerate(similar_blocks):
                variation = generate_variation_for_block(sim_bloco, dominio)
                st.write(f"Opção {i+1}: {variation}")
                if st.button(f"Aprovar Opção {i+1}", key=f"approve_{i}"):
                    add_new_block_from_similar(memoria, dominio, sim_bloco, variation)
                    st.success("Novo bloco salvo!")
                    st.rerun()
        else:
            st.write("Pensamento Divino: Nenhum bloco similar encontrado no conhecimento prévio para gerar variações. O sistema continua aprendendo.")
        st.rerun()

    # Botão Enter para gerar variações se há bloco atual e última entrada foi válida
    if st.session_state.current_bloco and st.session_state.last_valid:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Enter", key="enter_button"):
                texts = st.session_state.current_bloco["saidas"][0]["textos"]
                emoji = st.session_state.current_bloco["saidas"][0].get("reacao", "")
                
                # Gerar variações para cada texto, sem misturar textos
                all_variations = []
                for txt in texts:
                    variations = [txt]  # Sempre incluir o original
                    # Adicionar variações do inconsciente se existirem
                    bloco_inco = next((b for b in inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(st.session_state.current_bloco["bloco_id"])), None)
                    if bloco_inco:
                        for marker, data in bloco_inco["SAÍDA"].items():
                            if data["token"] in txt:
                                # Adicionar vars válidas (remover "0.0" que é placeholder)
                                valid_vars = [v for v in data["vars"] if v != "0.0"]
                                for var in valid_vars:
                                    # Substituir o token no texto pela var
                                    varied_txt = txt.replace(data["token"], var)
                                    variations.append(varied_txt)
                    all_variations.extend(variations)  # Adicionar variações deste texto
                
                # Adicionar Multivars_Saída
                multivars_saida = st.session_state.current_bloco["saidas"][0].get("Multivars_Saída", [])
                all_variations.extend(multivars_saida)
                
                # Escolher variação aleatoriamente com pesos
                bloco_id = str(st.session_state.current_bloco["bloco_id"])
                chosen = weighted_choice(all_variations, bloco_id)
                
                chosen = variar_texto(chosen, st.session_state.current_bloco, dominio)
                
                response = f"{chosen} {emoji}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Armazenar a última resposta para like
                st.session_state.last_response = chosen
                st.session_state.last_bloco_id = str(st.session_state.current_bloco["bloco_id"])
                st.session_state.last_button = "Enter"
                # Incrementar variação para próxima vez (removido, agora random)
                # Generate speech - sistema híbrido: Edge TTS para premium, gTTS para leves, pyttsx3 para outras
                if TTS_AVAILABLE and voz:
                    try:
                        if voz.startswith('edge-') and EDGE_TTS_AVAILABLE:
                            # Usar Edge TTS para vozes premium
                            voice_name = voz.split('-', 1)[1]
                            
                            edge_voice_map = {
                                'pt-br': 'pt-BR-FranciscaNeural',  # Feminina
                                'pt-pt': 'pt-PT-RaquelNeural',     # Feminina
                                'en': 'en-US-AriaNeural',          # Feminina
                                'en-us': 'en-US-AriaNeural',       # Feminina
                                'en-gb': 'en-GB-SoniaNeural',      # Feminina
                                'es': 'es-ES-ElviraNeural',        # Feminina
                                'es-us': 'es-US-PalomaNeural',     # Feminina
                                'fr': 'fr-FR-DeniseNeural',        # Feminina
                                'de': 'de-DE-KatjaNeural',         # Feminina
                                'it': 'it-IT-ElsaNeural',          # Feminina
                                'ja': 'ja-JP-NanamiNeural',        # Feminina
                                'ko': 'ko-KR-SunHiNeural',         # Feminina
                                'ru': 'ru-RU-SvetlanaNeural',      # Feminina
                                'ar': 'ar-SA-ZariyahNeural',       # Feminina
                                'hi': 'hi-IN-SwaraNeural',         # Feminina
                                'female': 'en-US-AriaNeural',      # Feminina
                                'pt-br-male': 'pt-BR-AntonioNeural',    # Masculina
                                'en-male': 'en-US-AndrewNeural',        # Masculina
                                'es-male': 'es-ES-AlvaroNeural',        # Masculina
                                'fr-male': 'fr-FR-HenriNeural',         # Masculina
                                'de-male': 'de-DE-ConradNeural',        # Masculina
                                'it-male': 'it-IT-DiegoNeural',         # Masculina
                                'ja-male': 'ja-JP-KeitaNeural',         # Masculina
                                'ko-male': 'ko-KR-InJoonNeural',        # Masculina
                                'ru-male': 'ru-RU-DmitryNeural',        # Masculina
                                'ar-male': 'ar-SA-HamedNeural',         # Masculina
                                'hi-male': 'hi-IN-MadhurNeural',        # Masculina
                                'male': 'en-US-ZiraNeural',             # Masculina (nota: Zira é feminino, mas usado como padrão masculino)
                            }
                            
                            selected_voice = edge_voice_map.get(voice_name, 'en-US-AriaNeural')
                            
                            import asyncio
                            import io
                            
                            async def generate_edge_audio():
                                communicate = edge_tts.Communicate(chosen, selected_voice)
                                audio_data = b""
                                async for chunk in communicate.stream():
                                    if chunk["type"] == "audio":
                                        audio_data += chunk["data"]
                                return audio_data
                            
                            audio_bytes = asyncio.run(generate_edge_audio())
                            
                            if audio_bytes and len(audio_bytes) > 0:
                                st.session_state.last_audio = audio_bytes
                                st.audio(st.session_state.last_audio, format='audio/mp3')
                                st.success(f"🎵 Áudio gerado com Edge TTS '{selected_voice}': {len(audio_bytes)} bytes")
                            else:
                                st.error("❌ Falha ao gerar arquivo de áudio com Edge TTS.")
                        elif voz.startswith('gtts-') and GTTS_AVAILABLE:
                            # Usar gTTS para vozes leves
                            lang_code = voz.split('-', 1)[1]
                            
                            lang_map = {
                                'pt-br': 'pt-br', 'pt-pt': 'pt-pt', 'en': 'en', 'en-us': 'en', 'en-gb': 'en',
                                'es': 'es', 'es-us': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'ja': 'ja',
                                'ko': 'ko', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
                            }
                            
                            if lang_code in lang_map:
                                from gtts import gTTS
                                import io
                                
                                tts = gTTS(text=chosen, lang=lang_map[lang_code], slow=False)
                                audio_buffer = io.BytesIO()
                                tts.write_to_fp(audio_buffer)
                                audio_buffer.seek(0)
                                audio_bytes = audio_buffer.read()
                                
                                if audio_bytes and len(audio_bytes) > 0:
                                    st.session_state.last_audio = audio_bytes
                                    st.audio(st.session_state.last_audio, format='audio/mp3')
                                    st.success(f"🎵 Áudio gerado com gTTS '{lang_code}': {len(audio_bytes)} bytes")
                                else:
                                    st.error("❌ Falha ao gerar arquivo de áudio com gTTS.")
                            else:
                                st.warning(f"Idioma '{lang_code}' não suportado pelo gTTS.")
                        else:
                            # Usar pyttsx3 para outras vozes
                            import pyttsx3
                            engine = pyttsx3.init()

                            voices = engine.getProperty('voices')
                            if voz.startswith('tortoise-'):
                                voice_name = voz.split('-', 1)[1]
                                if 'emma' in voice_name.lower() or 'female' in voice_name.lower():
                                    selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                else:
                                    selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                            else:
                                selected_voice = next((v for v in voices if v.name == voz), voices[0] if voices else None)

                            if not selected_voice and not voz.startswith('tortoise-'):
                                if genero == "masculino":
                                    selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                elif genero == "feminino":
                                    selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                else:
                                    selected_voice = random.choice(voices) if voices else None

                            if selected_voice:
                                engine.setProperty('voice', selected_voice.id)
                                engine.setProperty('rate', 180)
                                engine.setProperty('volume', 0.9)
                                engine.say(chosen)
                                engine.runAndWait()
                                st.success(f"🎵 Áudio reproduzido com sucesso! (Voz: {selected_voice.name})")
                            else:
                                st.warning("⚠️ Nenhuma voz do sistema encontrada.")

                    except Exception as e:
                        import traceback
                        st.error(f"Erro ao reproduzir áudio: {str(e)}")
                        st.error("Detalhes do erro:")
                        st.code(traceback.format_exc())
                        st.warning("TTS falhou, mas a conversa continua normalmente.")
        with col2:
            if st.button("💡 Insight", key="insight_button"):
                bloco = st.session_state.current_bloco
                insight_msg = generate_insight(bloco, st.session_state.get("last_response"))
                if insight_msg:
                    st.session_state.messages.append({"role": "assistant", "content": insight_msg})
                    with st.chat_message("assistant"):
                        st.markdown(insight_msg)
                    # Armazenar a última resposta para like (mas like só para Enter)
                    st.session_state.last_response = insight_msg
                    st.session_state.last_bloco_id = str(bloco["bloco_id"])
                    st.session_state.last_button = "Insight"
                    # Generate speech - sistema híbrido
                    if TTS_AVAILABLE:
                        try:
                            if voz and voz.startswith('edge-') and EDGE_TTS_AVAILABLE:
                                # Usar Edge TTS para vozes premium
                                voice_name = voz.split('-', 1)[1]
                                
                                edge_voice_map = {
                                    'pt-br': 'pt-BR-FranciscaNeural',  # Feminina
                                    'pt-pt': 'pt-PT-RaquelNeural',     # Feminina
                                    'en': 'en-US-AriaNeural',          # Feminina
                                    'en-us': 'en-US-AriaNeural',       # Feminina
                                    'en-gb': 'en-GB-SoniaNeural',      # Feminina
                                    'es': 'es-ES-ElviraNeural',        # Feminina
                                    'es-us': 'es-US-PalomaNeural',     # Feminina
                                    'fr': 'fr-FR-DeniseNeural',        # Feminina
                                    'de': 'de-DE-KatjaNeural',         # Feminina
                                    'it': 'it-IT-ElsaNeural',          # Feminina
                                    'ja': 'ja-JP-NanamiNeural',        # Feminina
                                    'ko': 'ko-KR-SunHiNeural',         # Feminina
                                    'ru': 'ru-RU-SvetlanaNeural',      # Feminina
                                    'ar': 'ar-SA-ZariyahNeural',       # Feminina
                                    'hi': 'hi-IN-SwaraNeural',         # Feminina
                                    'female': 'en-US-AriaNeural',      # Feminina
                                    'pt-br-male': 'pt-BR-AntonioNeural',    # Masculina
                                    'en-male': 'en-US-AndrewNeural',        # Masculina
                                    'es-male': 'es-ES-AlvaroNeural',        # Masculina
                                    'fr-male': 'fr-FR-HenriNeural',         # Masculina
                                    'de-male': 'de-DE-ConradNeural',        # Masculina
                                    'it-male': 'it-IT-DiegoNeural',         # Masculina
                                    'ja-male': 'ja-JP-KeitaNeural',         # Masculina
                                    'ko-male': 'ko-KR-InJoonNeural',        # Masculina
                                    'ru-male': 'ru-RU-DmitryNeural',        # Masculina
                                    'ar-male': 'ar-SA-HamedNeural',         # Masculina
                                    'hi-male': 'hi-IN-MadhurNeural',        # Masculina
                                    'male': 'en-US-ZiraNeural',             # Masculina (nota: Zira é feminino, mas usado como padrão masculino)
                                }
                                
                                selected_voice = edge_voice_map.get(voice_name, 'en-US-AriaNeural')
                                
                                import asyncio
                                import io
                                
                                async def generate_edge_audio():
                                    communicate = edge_tts.Communicate(insight_msg, selected_voice)
                                    audio_data = b""
                                    async for chunk in communicate.stream():
                                        if chunk["type"] == "audio":
                                            audio_data += chunk["data"]
                                    return audio_data
                                
                                audio_bytes = asyncio.run(generate_edge_audio())
                                
                                if audio_bytes and len(audio_bytes) > 0:
                                    st.session_state.last_audio = audio_bytes
                                    st.audio(st.session_state.last_audio, format='audio/mp3')
                                    st.success(f"🎵 Áudio gerado com Edge TTS '{selected_voice}': {len(audio_bytes)} bytes")
                                else:
                                    st.error("❌ Falha ao gerar arquivo de áudio com Edge TTS.")
                            elif voz and voz.startswith('gtts-') and GTTS_AVAILABLE:
                                # Usar gTTS para vozes leves
                                lang_code = voz.split('-', 1)[1]
                                
                                lang_map = {
                                    'pt-br': 'pt-br', 'pt-pt': 'pt-pt', 'en': 'en', 'en-us': 'en', 'en-gb': 'en',
                                    'es': 'es', 'es-us': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'ja': 'ja',
                                    'ko': 'ko', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi'
                                }
                                
                                if lang_code in lang_map:
                                    from gtts import gTTS
                                    import io
                                    
                                    tts = gTTS(text=insight_msg, lang=lang_map[lang_code], slow=False)
                                    audio_buffer = io.BytesIO()
                                    tts.write_to_fp(audio_buffer)
                                    audio_buffer.seek(0)
                                    audio_bytes = audio_buffer.read()
                                    
                                    if audio_bytes and len(audio_bytes) > 0:
                                        st.session_state.last_audio = audio_bytes
                                        st.audio(st.session_state.last_audio, format='audio/mp3')
                                        st.success(f"🎵 Áudio gerado com gTTS '{lang_code}': {len(audio_bytes)} bytes")
                                    else:
                                        st.error("❌ Falha ao gerar arquivo de áudio com gTTS.")
                                else:
                                    st.warning(f"Idioma '{lang_code}' não suportado pelo gTTS.")
                            else:
                                # Usar pyttsx3 para outras vozes
                                import pyttsx3
                                engine = pyttsx3.init()

                                voices = engine.getProperty('voices')
                                if voz.startswith('tortoise-'):
                                    voice_name = voz.split('-', 1)[1]
                                    if 'emma' in voice_name.lower() or 'female' in voice_name.lower():
                                        selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                    else:
                                        selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                else:
                                    selected_voice = next((v for v in voices if v.name == voz), voices[0] if voices else None)

                                if not selected_voice and not voz.startswith('tortoise-'):
                                    if genero == "masculino":
                                        selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['david', 'mark', 'male', 'paul', 'george'])), voices[0] if voices else None)
                                    elif genero == "feminino":
                                        selected_voice = next((v for v in voices if any(k in v.name.lower() for k in ['maria', 'zira', 'hazel', 'female', 'anna', 'linda'])), voices[0] if voices else None)
                                    else:
                                        selected_voice = random.choice(voices) if voices else None

                                if selected_voice:
                                    engine.setProperty('voice', selected_voice.id)
                                    engine.setProperty('rate', 180)
                                    engine.setProperty('volume', 0.9)
                                    engine.say(insight_msg)
                                    engine.runAndWait()
                                    st.success(f"🎵 Áudio reproduzido com sucesso! (Voz: {selected_voice.name})")
                                else:
                                    st.warning("⚠️ Nenhuma voz do sistema encontrada.")
                        except Exception as e:
                            import traceback
                            st.error(f"Erro ao reproduzir áudio: {str(e)}")
                            st.error("Detalhes do erro:")
                            st.code(traceback.format_exc())
                            st.warning("TTS falhou, mas a conversa continua normalmente.")

        st.rerun()

    # Botão de Like se há última resposta do Enter
    if "last_response" in st.session_state and "last_bloco_id" in st.session_state and st.session_state.get("last_button") == "Enter":
        if st.button("👍 Like na última resposta"):
            bloco_id = st.session_state.last_bloco_id
            response = st.session_state.last_response
            if bloco_id not in st.session_state.likes:
                st.session_state.likes[bloco_id] = {}
            if response not in st.session_state.likes[bloco_id]:
                st.session_state.likes[bloco_id][response] = 0
            st.session_state.likes[bloco_id][response] += 1
            st.success(f"👍 Curtido! Agora '{response}' tem mais chances de aparecer.")
            st.rerun()


def weighted_choice(variations, bloco_id):
    """Escolhe uma variação com pesos baseados em likes."""
    if bloco_id not in st.session_state.likes:
        st.session_state.likes[bloco_id] = {}
    weights = []
    for var in variations:
        count = st.session_state.likes[bloco_id].get(var, 0)
        weights.append(max(1, count + 1))  # mínimo 1 para não zerar
    return random.choices(variations, weights=weights, k=1)[0]


def get_bloco_from_text(entrada: str, dominio: str) -> dict:
    memoria = st.session_state.memoria
    if dominio not in memoria["IM"]:
        return None
    blocos = memoria["IM"][dominio]["blocos"]
    for bloco in blocos:
        if bloco["entrada"]["texto"] == entrada:
            return bloco
    return None


def get_unconscious_vars_for_block(bloco: dict, dominio: str) -> dict:
    inconsciente = st.session_state.inconsciente
    if dominio not in inconsciente.get("INCO", {}):
        return {}
    blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
    bloco_inco = next((b for b in blocos_inco if b["Bloco_id"] == str(bloco["bloco_id"])), None)
    if not bloco_inco:
        return {}
    vars_dict = {}
    for data in bloco_inco["Entrada"].values():
        token = data["token"]
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            vars_dict[token] = vars_list
    for data in bloco_inco["SAÍDA"].values():
        token = data["token"]
        vars_list = data["vars"]
        if vars_list and vars_list != ["0.0"]:
            vars_dict[token] = vars_list
    return vars_dict


def generate_cartesian_responses(texts: list, unconscious_vars_dict: dict) -> list:
    # Para frases "Olá [A] [B]."
    a_words = set()
    b_words = set()
    for text in texts:
        tokens = Token(text)
        if len(tokens) > 1:
            a_words.add(tokens[1])
        if len(tokens) > 2:
            b_words.add(tokens[2])
    a_variations = list(a_words)
    for word in a_words:
        if word in unconscious_vars_dict:
            a_variations.extend(unconscious_vars_dict[word])
    b_variations = list(b_words)
    for word in b_words:
        if word in unconscious_vars_dict:
            b_variations.extend(unconscious_vars_dict[word])
    b_variations = [""] + b_variations  # incluir vazio
    combinations = []
    for a in a_variations:
        for b in b_variations:
            if b:
                combinations.append(f"Olá {a} {b}.")
            else:
                combinations.append(f"Olá {a}.")
    return combinations


def atualizar_inconsciente_para_im(memoria: dict, dominio: str) -> None:
    """Atualiza o inconsciente para o IM selecionado."""
    if dominio not in memoria["IM"]:
        return
    im_data = memoria["IM"][dominio]
    inconsciente = st.session_state.inconsciente
    if dominio not in inconsciente.get("INCO", {}):
        inconsciente.setdefault("INCO", {})[dominio] = {
            "NOME": im_data.get("nome", f"IM_{dominio}"),
            "Ultimo child": im_data.get("ultimo_child", f"{dominio}.0"),
            "Blocos": []
        }
        salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def test_model(memoria: dict, dominio: str) -> None:
    # Atualizar inconsciente para o IM selecionado
    atualizar_inconsciente_para_im(memoria, dominio)

    ckpt = ckpt_path(dominio)
    if not os.path.exists(ckpt):
        st.warning("⚠️ Sem checkpoint — treine primeiro.");
        return

    (state,
     maxE, maxRE, maxCE, maxPIDE,
     mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
     vE, vRE, vCE, vPIDE,
     n_txt, n_emo, n_ctx,
     max_ng
     ) = torch.load(ckpt)

    model = AdamSegmentado(
        nE=len(vE), nRE=len(vRE), nCE=len(vCE), nPIDE=len(vPIDE),
        mom_size=mom_size,
        num_vals_E=len(val_to_idx_E), num_vals_RE=len(val_to_idx_RE),
        num_vals_CE=len(val_to_idx_CE), num_vals_PIDE=len(val_to_idx_PIDE),
        n_txt=n_txt, n_emo=n_emo,
        n_ctx=n_ctx,
        max_E=maxE, max_RE=maxRE, max_CE=maxCE, max_PIDE=maxPIDE, max_ng=max_ng
    )
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        st.warning(f"⚠️ Checkpoint incompatível devido a mudanças na arquitetura: {e}. Treine primeiro.")
        return
    model.eval()

    blocos = memoria["IM"][dominio]["blocos"]
    inconsciente = st.session_state.inconsciente
    ultimo_child_per_block = {}
    if dominio in inconsciente.get("INCO", {}):
        blocos_inco = inconsciente["INCO"][dominio].get("Blocos", [])
        for bloco in blocos_inco:
            bloco_num = int(bloco["Bloco_id"])
            saida_vals = [float(key) for key in bloco.get("SAÍDA", {}).keys()]
            if saida_vals:
                ultimo_child_per_block[bloco_num] = max(saida_vals)
            else:
                ultimo_child_per_block[bloco_num] = 0.50
    st.write(f"📊 Teste em lote — Domínio {dominio} ({len(blocos)} blocos)")

    # Inicializar acumuladores para métricas
    total_samples = 0
    acc_txt = 0.0
    acc_emo = 0.0
    acc_ctx = 0.0
    mse_pos = 0.0

    for b in blocos:
        max_val = ultimo_child_per_block.get(b["bloco_id"], 0.50)

        def featurize(field, max_len, vocab, val_to_idx, max_ng):
            tokens = b["entrada"]["tokens"].get(field, [])
            ngrams_list = [generate_ngrams(t, N_GRAM) for t in tokens]
            ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
            val_idxs = [val_to_idx.get(t, 0) for t in tokens]
            vals = [float(t) for t in tokens]
            moms = [int(t.split(".", 1)[0]) for t in tokens]
            if vals:
                min_v, max_v = min(vals), max(vals)
                pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
            else:
                pos = []
            pad_ids = (max_len * max_ng) - len(ids)
            pad_vals = max_len - len(tokens)
            ids += [0] * pad_ids
            val_idxs += [0] * pad_vals
            vals += [0.0] * pad_vals
            moms += [0] * pad_vals
            pos += [0.0] * pad_vals
            return (
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([val_idxs], dtype=torch.long),
                torch.tensor([vals], dtype=torch.float32),
                torch.tensor([moms], dtype=torch.long),
                torch.tensor([pos], dtype=torch.float32),
            )

        E_ids, E_val_idxs, E_val, E_mom, E_pos = featurize("E", maxE, vE, val_to_idx_E, max_ng)
        RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos = featurize("RE", maxRE, vRE, val_to_idx_RE, max_ng)
        CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE", maxCE, vCE, val_to_idx_CE, max_ng)
        PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

        x = {
            "E": E_ids, "E_val": E_val, "E_mom": E_mom, "E_pos": E_pos, "E_val_idx": E_val_idxs,
            "RE": RE_ids, "RE_val": RE_val, "RE_mom": RE_mom, "RE_pos": RE_pos, "RE_val_idx": RE_val_idxs,
            "CE": CE_ids, "CE_val": CE_val, "CE_mom": CE_mom, "CE_pos": CE_pos, "CE_val_idx": CE_val_idxs,
            "PIDE": PI_ids, "PIDE_val": PI_val, "PIDE_mom": PI_mom, "PIDE_pos": PI_pos, "PIDE_val_idx": PI_val_idxs,
        }

        with torch.no_grad():
            out = model(x)

        # Calcular métricas para este bloco
        pred_txt = out["texto"].argmax(dim=1).item()
        pred_emo = out["emoji"].argmax(dim=1).item()
        pred_ctx = out["ctx"].argmax(dim=1).item()
        pred_pos = out["pos"].item()

        true_texts = [normalize(t) for t in b["saidas"][0]["textos"]]
        pred_text = b["saidas"][0]["textos"][pred_txt] if pred_txt < len(b["saidas"][0]["textos"]) else "N/A"
        true_emo = b["saidas"][0].get("reacao", "")
        true_ctx = b["saidas"][0].get("contexto", "")
        # Para pos, usar a média dos valores dos tokens do bloco
        all_vals = []
        for field in ["E", "RE", "CE", "PIDE"]:
            tokens = b["entrada"]["tokens"].get(field, [])
            all_vals.extend([float(t) for t in tokens])
        true_pos = sum(all_vals) / len(all_vals) if all_vals else 0.0

        # Acurácias (comparar índices)
        acc_txt_block = 1 if normalize(pred_text) in true_texts else 0
        acc_emo_block = 1 if pred_emo == 0 else 0  # 0 correto
        acc_ctx_block = 1 if pred_ctx == 0 else 0
        mse_pos_block = (pred_pos - true_pos) ** 2

        acc_txt += acc_txt_block
        acc_emo += acc_emo_block
        acc_ctx += acc_ctx_block
        mse_pos += mse_pos_block

        total_samples += 1

        # Coletar valores únicos no bloco
        block_vals = set()
        for field in ["E", "RE", "CE", "PIDE"]:
            block_vals.update(float(t) for t in b["entrada"]["tokens"].get(field, []) if t)

        if st.session_state.get("admin", False):
            st.write(f"\n❏ Bloco_id={b['bloco_id']} Entrada: {b['entrada']['texto']} {b['entrada']['reacao']}")
            st.write(f"   Texto pred: {pred_text} | True: {true_texts}")
            st.write(f"   Emoji pred: {true_emo if pred_emo == 0 else 'Outro'} | True: {true_emo}")
            st.write(f"   Contexto pred: {true_ctx if pred_ctx == 0 else 'Outro'} | True: {true_ctx}")
            st.write(f"   Posição pred: {pred_pos:.4f} | True: {true_pos:.4f}")
            st.write(f"   Acurácia Texto: {acc_txt_block:.1f}")
            st.write(f"   Acurácia Emoji: {acc_emo_block:.1f}")
            st.write(f"   Acurácia Contexto: {acc_ctx_block:.1f}")
            st.write(f"   MSE Posição: {mse_pos_block:.4f}")

    # Calcular médias
    if total_samples > 0:
        acc_txt /= total_samples
        acc_emo /= total_samples
        acc_ctx /= total_samples
        mse_pos /= total_samples

        if not st.session_state.get("admin", False):
            st.info("📋 Detalhes dos testes disponíveis apenas para administradores. As métricas gerais são exibidas abaixo.")

        st.write("\n📈 Métricas Gerais:")
        st.write(f"Acurácia Texto: {acc_txt:.2%}")
        st.write(f"Acurácia Emoji: {acc_emo:.2%}")
        st.write(f"Acurácia Contexto: {acc_ctx:.2%}")
        st.write(f"MSE Posição: {mse_pos:.4f}")
    else:
        st.write("Nenhum bloco para testar.")


## INSEPA_CLI
def prompt_dominio(action: str, memoria: dict) -> str:
    """Lista IMs disponíveis e permite escolher um para a ação especificada."""
    ims = list(memoria.get("IM", {}).keys())
    if not ims:
        st.error("❌ Nenhum IM encontrado. Crie um primeiro.")
        return ""

    if action == "conversar":
        st.write("Selecione o Universo para conversar:")
    else:
        st.write(f"\n--- Escolher IM para {action} ---")
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            num_blocos = len(memoria["IM"][im_id].get("blocos", []))
            st.write(f"- {im_id}: {nome} ({num_blocos} blocos)")

    dom = st.selectbox(f"Escolha o Universo para {action}", ims, key=f"dominio_{action}")
    return dom


def create_new_im(memoria: dict) -> None:
    """Cria um novo IM (Índice Mãe) vazio."""
    im_id = st.text_input("Índice mãe para o novo IM:", key="new_im_id")
    if not im_id.isdigit():
        st.error("❌ Índice mãe deve ser um número.")
        return
    im_id = int(im_id)
    if str(im_id) in memoria.get("IM", {}):
        st.error(f"❌ IM {im_id} já existe.")
        return
    nome = st.text_input("Nome do IM (opcional):", key="new_im_name") or f"IM_{im_id}"
    genero = st.selectbox("Gênero do IM:", ["masculino", "feminino", "não binário", "outro"], key="new_im_genero")
    voz = None
    if TTS_AVAILABLE:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        gtts_voices = ['gtts-pt-br', 'gtts-pt-pt', 'gtts-en', 'gtts-en-us', 'gtts-en-gb', 'gtts-es', 'gtts-es-us', 'gtts-fr', 'gtts-de', 'gtts-it', 'gtts-ja', 'gtts-ko', 'gtts-ru', 'gtts-ar', 'gtts-hi']
        coqui_voices = ['tts_models/pt/cv/vits', 'tts_models/en/ljspeech/tacotron2-DDC_ph']
        voice_options = [v.name for v in voices if v] + gtts_voices + [f"coqui-{cv}" for cv in coqui_voices]
        voz = st.selectbox("Voz preferida (opcional):", ["Automático"] + voice_options, key="new_im_voz")
        if voz == "Automático":
            voz = None
    if st.button("Criar IM"):
        im_data = {
            "nome": nome,
            "genero": genero,
            "ultimo_child": f"{im_id}.0",
            "blocos": []
        }
        if voz:
            im_data["voz"] = voz
        memoria.setdefault("IM", {})[str(im_id)] = im_data
        salvar_json(ARQUIVO_MEMORIA, memoria)
        st.success(f"✅ IM {im_id} criado: {nome} ({genero})" + (f" - Voz: {voz}" if voz else ""))


def submenu_im(memoria: dict, inconsciente: dict) -> None:
    st.subheader("🛠️ Gerenciar IMs e Blocos")
    st.write("Áudio disponível. Ouça a voz do personagem escolhido agora")
    st.write(f"gTTS: {GTTS_AVAILABLE}")
    st.write(f"Python executable: {sys.executable}")
    sub_opc = st.selectbox("Escolha uma opção:", [
        "📋 Visualizar IMs e Blocos",
        "➕ Criar novo IM",
        "🔧 Gerar bloco a partir de template INSEPA",
        "🗑️ Apagar bloco",
        "🚮 Apagar IM",
        "⚙️ Alimentar vars dos tokens",
        "✏️ Editar nomes de IMs",
        "💾 Backup JSON",
        "⬅️ Voltar ao menu principal"
    ], key="submenu_im")

    if sub_opc == "📋 Visualizar IMs e Blocos":
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.info("Nenhum IM encontrado. Crie um primeiro.")
            return
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            genero = memoria["IM"][im_id].get("genero", "não definido")
            voz = memoria["IM"][im_id].get("voz", None)
            num_blocos = len(memoria["IM"][im_id].get("blocos", []))
            with st.expander(f"📁 IM {im_id}: {nome} ({genero})" + (f" - Voz: {voz}" if voz else "") + f" ({num_blocos} blocos)"):
                blocos = memoria["IM"][im_id].get("blocos", [])
                if blocos:
                    # Tabela de Entrada
                    data_entrada = [
                        {
                            "ID": b["bloco_id"],
                            "Entrada": b["entrada"]["texto"],
                            "Reação": b["entrada"].get("reacao", ""),
                            "Contexto": b["entrada"].get("contexto", ""),
                            "Pensamento Interno": b["entrada"].get("pensamento_interno", "")
                        } for b in blocos
                    ]
                    st.subheader("📥 Entradas dos Blocos")
                    st.dataframe(data_entrada, use_container_width=True, column_config={
                        "Entrada": st.column_config.TextColumn("Entrada", width=None),
                        "Reação": st.column_config.TextColumn("Reação", width=None),
                        "Contexto": st.column_config.TextColumn("Contexto", width=None),
                        "Pensamento Interno": st.column_config.TextColumn("Pensamento Interno", width=None)
                    })
                    
                    # Multivars de Entrada
                    st.subheader("🔄 Multivars de Entrada (Frases Completas)")
                    multivars_entrada_data = [
                        {
                            "ID": b["bloco_id"],
                            "Multivars_Entrada": "\n".join(b["entrada"].get("Multivars_Entrada", [])) or "Nenhum"
                        } for b in blocos
                    ]
                    st.dataframe(multivars_entrada_data, use_container_width=True, column_config={
                        "Multivars_Entrada": st.column_config.TextColumn("Multivars_Entrada", width=None)
                    })
                    
                    # Tabela de Saída
                    data_saida = [
                        {
                            "ID": b["bloco_id"],
                            "Saídas": "\n".join(b["saidas"][0]["textos"]),
                            "Reação": b["saidas"][0].get("reacao", ""),
                            "Contexto": b["saidas"][0].get("contexto", "")
                        } for b in blocos
                    ]
                    st.subheader("📤 Saídas dos Blocos")
                    st.dataframe(data_saida, use_container_width=True, column_config={
                        "Saídas": st.column_config.TextColumn("Saídas", width=None),
                        "Reação": st.column_config.TextColumn("Reação", width=None),
                        "Contexto": st.column_config.TextColumn("Contexto", width=None)
                    })
                    
                    # Multivars de Saída
                    st.subheader("🔄 Multivars de Saída (Frases Completas)")
                    multivars_saida_data = [
                        {
                            "ID": b["bloco_id"],
                            "Multivars_Saída": "\n".join(b["saidas"][0].get("Multivars_Saída", [])) or "Nenhum"
                        } for b in blocos
                    ]
                    st.dataframe(multivars_saida_data, use_container_width=True, column_config={
                        "Multivars_Saída": st.column_config.TextColumn("Multivars_Saída", width=None)
                    })
                    
                    # Lista de vozes disponíveis
                    if TTS_AVAILABLE:
                        st.subheader("🎤 Vozes Disponíveis para TTS")
                        
                        st.write("**Vozes do Google Text-to-Speech (gTTS):**")
                        if GTTS_AVAILABLE:
                            gtts_voices = [
                                "gtts-pt-br (Português Brasil)", "gtts-pt-pt (Português Portugal)", 
                                "gtts-en (Inglês)", "gtts-en-us (Inglês EUA)", "gtts-en-gb (Inglês GB)",
                                "gtts-es (Espanhol)", "gtts-es-us (Espanhol EUA)", "gtts-fr (Francês)",
                                "gtts-de (Alemão)", "gtts-it (Italiano)", "gtts-ja (Japonês)",
                                "gtts-ko (Coreano)", "gtts-ru (Russo)", "gtts-ar (Árabe)", "gtts-hi (Hindi)"
                            ]
                            for voice in gtts_voices:
                                st.write(f"- {voice}")
                        else:
                            st.write("- gTTS não disponível")
                        
                        st.write("**Vozes do Edge TTS (Microsoft Edge - Premium):**")
                        if EDGE_TTS_AVAILABLE:
                            edge_voices = [
                                "edge-pt-br (Português Brasil - Francisca)", "edge-pt-br-male (Português Brasil - Antonio)",
                                "edge-en (Inglês EUA - Aria)", "edge-en-male (Inglês EUA - Andrew)",
                                "edge-es (Espanhol - Elvira)", "edge-fr (Francês - Denise)",
                                "edge-de (Alemão - Katja)", "edge-it (Italiano - Elsa)",
                                "edge-ja (Japonês - Nanami)", "edge-ko (Coreano - SunHi)",
                                "edge-ru (Russo - Svetlana)", "edge-ar (Árabe - Zariyah)",
                                "edge-hi (Hindi - Hemant)"
                            ]
                            for voice in edge_voices:
                                st.write(f"- {voice}")
                        else:
                            st.write("- Edge TTS não disponível")
                        
                        # Alterar voz do IM
                        st.subheader("🎤 Alterar Voz do IM")
                        voz_atual = memoria["IM"][im_id].get("voz", None)
                        
                        # Mapeamento de códigos para nomes descritivos
                        code_to_name = {
                            # Google TTS
                            "gtts-pt-br": "Google TTS - Português Brasil",
                            "gtts-pt-pt": "Google TTS - Português Portugal",
                            "gtts-en": "Google TTS - Inglês",
                            "gtts-en-us": "Google TTS - Inglês (EUA)",
                            "gtts-en-gb": "Google TTS - Inglês (GB)",
                            "gtts-es": "Google TTS - Espanhol",
                            "gtts-es-us": "Google TTS - Espanhol (EUA)",
                            "gtts-fr": "Google TTS - Francês",
                            "gtts-de": "Google TTS - Alemão",
                            "gtts-it": "Google TTS - Italiano",
                            "gtts-ja": "Google TTS - Japonês",
                            "gtts-ko": "Google TTS - Coreano",
                            "gtts-ru": "Google TTS - Russo",
                            "gtts-ar": "Google TTS - Árabe",
                            "gtts-hi": "Google TTS - Hindi",
                            # Edge TTS
                            "edge-pt-br": "Edge TTS - Português Brasil (Francisca - Feminina)",
                            "edge-pt-br-male": "Edge TTS - Português Brasil (Antônio - Masculino)",
                            "edge-en": "Edge TTS - Inglês (Jenny - Feminina)",
                            "edge-en-male": "Edge TTS - Inglês (Guy - Masculino)",
                            "edge-es": "Edge TTS - Espanhol (Helena - Feminina)",
                            "edge-fr": "Edge TTS - Francês (Denise - Feminina)",
                            "edge-de": "Edge TTS - Alemão (Katja - Feminina)",
                            "edge-it": "Edge TTS - Italiano (Elsa - Feminina)",
                            "edge-ja": "Edge TTS - Japonês (Nanami - Feminina)",
                            "edge-ko": "Edge TTS - Coreano (SunHi - Feminina)",
                            "edge-ru": "Edge TTS - Russo (Svetlana - Feminina)",
                            "edge-ar": "Edge TTS - Árabe (Hoda - Feminina)",
                            "edge-hi": "Edge TTS - Hindi (Hemant - Masculino)"
                        }
                        name_to_code = {v: k for k, v in code_to_name.items()}
                        
                        # Opções de voz: Automático e vozes com nomes descritivos
                        voz_options = ["Automático"]
                        
                        # Adicionar vozes do gTTS (Google Text-to-Speech)
                        if GTTS_AVAILABLE:
                            gtts_voices = [
                                "gtts-pt-br", "gtts-pt-pt", "gtts-en", "gtts-en-us", "gtts-en-gb", 
                                "gtts-es", "gtts-es-us", "gtts-fr", "gtts-de", "gtts-it", "gtts-ja", 
                                "gtts-ko", "gtts-ru", "gtts-ar", "gtts-hi"
                            ]
                            voz_options.extend([code_to_name[code] for code in gtts_voices if code in code_to_name])
                        
                        # Adicionar vozes do Edge TTS (Microsoft Edge)
                        if EDGE_TTS_AVAILABLE:
                            edge_voices = [
                                "edge-pt-br", "edge-pt-br-male", "edge-en", "edge-en-male", 
                                "edge-es", "edge-fr", "edge-de", "edge-it", "edge-ja", 
                                "edge-ko", "edge-ru", "edge-ar", "edge-hi"
                            ]
                            voz_options.extend([code_to_name[code] for code in edge_voices if code in code_to_name])
                        
                        default_index = 0
                        if voz_atual and voz_atual in code_to_name:
                            voz_nome_atual = code_to_name[voz_atual]
                            if voz_nome_atual in voz_options:
                                default_index = voz_options.index(voz_nome_atual)
                        voz_selecionada = st.selectbox("Selecione uma voz:", voz_options, index=default_index, key=f"voz_{im_id}")
                        if st.button("Salvar Voz", key=f"save_voz_{im_id}"):
                            if voz_selecionada == "Automático":
                                memoria["IM"][im_id].pop("voz", None)
                            else:
                                voz_code = name_to_code[voz_selecionada]
                                memoria["IM"][im_id]["voz"] = voz_code
                            salvar_json(ARQUIVO_MEMORIA, memoria)
                            st.success(f"✅ Voz do IM {im_id} atualizada para {voz_selecionada}!")
                            st.rerun()
                        
                        st.info("💡 **Sistema TTS Otimizado!** Edge TTS para vozes premium, gTTS para vozes leves e pyttsx3 como fallback. Sem Tortoise para melhor performance!")
                    
                    # Submenu para editar blocos
                    bloco_options = {f"ID {b['bloco_id']}: {b['entrada']['texto']}": b for b in blocos}
                    bloco_selecionado = st.selectbox("Selecione o bloco para editar:", list(bloco_options.keys()), key=f"edit_{im_id}")
                    bloco = bloco_options[bloco_selecionado]
                    with st.form(f"edit_bloco_{im_id}_{bloco['bloco_id']}"):
                        st.subheader("Editar Bloco")
                        entrada_texto = st.text_area("Entrada:", bloco["entrada"]["texto"], height=100)
                        multivars_entrada = st.text_area("Multivars Entrada (uma por linha):", "\n".join(bloco["entrada"].get("Multivars_Entrada", [])), height=100)
                        entrada_reacao = st.text_input("Reação (Entrada):", bloco["entrada"].get("reacao", ""))
                        entrada_contexto = st.text_area("Contexto (Entrada):", bloco["entrada"].get("contexto", ""), height=100)
                        entrada_pensamento = st.text_area("Pensamento Interno:", bloco["entrada"].get("pensamento_interno", ""), height=100)
                        saida_textos = st.text_area("Saída:", "\n".join(bloco["saidas"][0]["textos"]), height=150)
                        multivars_saida = st.text_area("Multivars Saída (uma por linha):", "\n".join(bloco["saidas"][0].get("Multivars_Saída", [])), height=100)
                        saida_reacao = st.text_input("Reação (Saída):", bloco["saidas"][0].get("reacao", ""))
                        saida_contexto = st.text_area("Contexto (Saída):", bloco["saidas"][0].get("contexto", ""), height=100)
                        if st.form_submit_button("Salvar Edições"):
                            bloco["entrada"] = {
                                "texto": entrada_texto,
                                "Multivars_Entrada": [m.strip() for m in multivars_entrada.split("\n") if m.strip()],
                                "reacao": entrada_reacao,
                                "contexto": entrada_contexto,
                                "pensamento_interno": entrada_pensamento,
                                "tokens": bloco["entrada"]["tokens"],
                                "fim": bloco["entrada"]["fim"],
                                "alnulu": bloco["entrada"]["alnulu"]
                            }
                            bloco["saidas"][0] = {
                                "textos": saida_textos.split("\n"),
                                "Multivars_Saída": [m.strip() for m in multivars_saida.split("\n") if m.strip()],
                                "reacao": saida_reacao,
                                "contexto": saida_contexto,
                                "tokens": bloco["saidas"][0]["tokens"],
                                "fim": bloco["saidas"][0]["fim"]
                            }
                            salvar_json(ARQUIVO_MEMORIA, memoria)
                            recalcular_marcadores_im(memoria, im_id)
                            st.success("Bloco editado com sucesso!")
                else:
                    st.write("Nenhum bloco.")
    elif sub_opc == "➕ Criar novo IM":
        create_new_im(memoria)
    elif sub_opc == "🔧 Gerar bloco a partir de template INSEPA":
        # Listar IMs existentes
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado. Crie um primeiro.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_gerar")
        st.write(f"Gerando blocos no IM {im_escolhido} (ou em outros se especificado no template)...")
        st.write("Cole seus blocos templates INSEPA separados por --- (cada um pode ter seu próprio 'Índice mãe:' ou será usado o selecionado acima):")
        template_text = st.text_area("Templates:", key="template_text", height=300)
        if st.button("Gerar Blocos"):
            blocks = template_text.split("---")
            generated_count = 0
            for block in blocks:
                block = block.strip()
                if block:
                    if not block.startswith("Índice mãe:"):
                        block = f"Índice mãe: {im_escolhido}\n" + block
                    try:
                        generate_block_from_template(memoria, block)
                        generated_count += 1
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar bloco: {e}")
            if generated_count > 0:
                st.success(f"✅ {generated_count} bloco(s) gerado(s) com sucesso!")
    elif sub_opc == "🗑️ Apagar bloco":
        # Apagar bloco
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_apagar_bloco")
        universo = memoria["IM"][im_escolhido]
        blocos = universo.get("blocos", [])
        if not blocos:
            st.error("❌ Nenhum bloco neste IM.")
            return
        st.write("Blocos no IM:")
        bloco_options = [f"ID {b['bloco_id']}: {b['entrada']['texto']}" for b in blocos]
        bloco_selecionado = st.selectbox("Selecione o bloco para apagar:", bloco_options, key="bloco_apagar")
        bid_apagar = bloco_selecionado.split(":")[0].split()[1]
        if st.button("Apagar Bloco"):
            try:
                bid_int = int(bid_apagar)
                bloco = next((b for b in blocos if b["bloco_id"] == bid_int), None)
                if bloco:
                    blocos.remove(bloco)
                    # Renumerar blocos sequencialmente
                    for i, b in enumerate(blocos, 1):
                        b["bloco_id"] = i
                    # Recalcular ultimo_child
                    if blocos:
                        universo["ultimo_child"] = max(b["saidas"][0]["fim"] for b in blocos)
                    else:
                        universo["ultimo_child"] = f"{im_escolhido}.0"
                    salvar_json(ARQUIVO_MEMORIA, memoria)

                    # Remover do inconsciente.json
                    inconsciente = st.session_state.inconsciente
                    if im_escolhido in inconsciente.get("INCO", {}):
                        blocos_inco = inconsciente["INCO"][im_escolhido].get("Blocos", [])
                        inconsciente["INCO"][im_escolhido]["Blocos"] = [b for b in blocos_inco if b["Bloco_id"] != str(bid_int)]
                        # Recalcular Ultimo child se necessário
                        if blocos:
                            saida_vals = []
                            for b in blocos:
                                if im_escolhido in inconsciente.get("INCO", {}) and "Blocos" in inconsciente["INCO"][im_escolhido]:
                                    bloco_inco = next((bi for bi in inconsciente["INCO"][im_escolhido]["Blocos"] if bi["Bloco_id"] == str(b["bloco_id"])), None)
                                    if bloco_inco:
                                        saida_vals.extend(float(k) for k in bloco_inco.get("SAÍDA", {}).keys())
                            if saida_vals:
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = str(max(saida_vals))
                            else:
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                        else:
                            if im_escolhido in inconsciente.get("INCO", {}):
                                inconsciente["INCO"][im_escolhido]["Ultimo child"] = f"{im_escolhido}.0"
                    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                    st.success(f"✅ Bloco {bid_int} apagado. Blocos renumerados e ultimo_child ajustado.")
                else:
                    st.error("❌ Bloco não encontrado.")
            except ValueError:
                st.error("❌ ID inválido.")
    elif sub_opc == "🚮 Apagar IM":
        # Apagar IM
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_apagar = st.selectbox("Digite o ID do IM:", ims, key="im_apagar")
        confirm = st.checkbox("Tem certeza que quer apagar o IM e todos os seus blocos?")
        if confirm and st.button("Apagar IM"):
            # Coletar blocos para remover do inconsciente
            blocos_a_remover = [f"Bloco_{b['bloco_id']}" for b in memoria["IM"][im_apagar]["blocos"]]
            del memoria["IM"][im_apagar]
            salvar_json(ARQUIVO_MEMORIA, memoria)

            # Remover do inconsciente.json
            inconsciente = st.session_state.inconsciente
            if im_apagar in inconsciente.get("INCO", {}):
                del inconsciente["INCO"][im_apagar]
            salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)

            st.success(f"✅ IM {im_apagar} apagado.")
    elif sub_opc == "⚙️ Alimentar vars dos tokens":
        # Alimentar vars dos tokens
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.error("❌ Nenhum IM encontrado.")
            return
        st.write("IMs disponíveis:")
        for im_id in ims:
            nome = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            st.write(f"- {im_id}: {nome}")
        im_escolhido = st.selectbox("Digite o ID do IM:", ims, key="im_escolhido_vars")
        inconsciente = st.session_state.inconsciente
        st.write(f"DEBUG: Inconsciente carregado. Chaves INCO: {list(inconsciente.get('INCO', {}).keys())}")
        if im_escolhido not in inconsciente.get("INCO", {}):
            st.error("❌ Nenhum bloco no inconsciente para este IM.")
            return
        im_data = inconsciente["INCO"][im_escolhido]
        st.write(f"DEBUG: IM {im_escolhido} tem {len(im_data.get('Blocos', []))} blocos")
        blocos = im_data.get("Blocos", [])
        if not blocos:
            st.error("❌ Nenhum bloco no inconsciente para este IM.")
            return
        st.write(f"Blocos do IM {im_escolhido}:")
        for bloco in blocos:
            with st.expander(f"Bloco {bloco['Bloco_id']}"):
                st.subheader("Entrada")
                st.write(f"DEBUG: Bloco tem {len(bloco['Entrada'])} entradas")
                for marker, data in bloco["Entrada"].items():
                    vars_list = data.get('vars', [])
                    st.write(f"{marker}: {data['token']} | vars: {vars_list}")
                    if vars_list and any(v != "0.0" for v in vars_list):
                        st.write(f"  ✅ Vars não vazias encontradas: {vars_list}")
                st.subheader("SAÍDA")
                st.write(f"DEBUG: Bloco tem {len(bloco['SAÍDA'])} saídas")
                for marker, data in bloco["SAÍDA"].items():
                    vars_list = data.get('vars', [])
                    st.write(f"{marker}: {data['token']} | vars: {vars_list}")
                    if vars_list and any(v != "0.0" for v in vars_list):
                        st.write(f"  ✅ Vars não vazias encontradas: {vars_list}")
        # Editar vars
        bloco_ids = [b["Bloco_id"] for b in blocos]
        bloco_edit = st.selectbox("Escolha o bloco para editar:", bloco_ids, key="bloco_edit")
        bloco = next((b for b in blocos if b["Bloco_id"] == bloco_edit), None)
        if bloco:
            campo_opc = st.selectbox("Escolha o campo:", ["Entrada", "SAÍDA"], key="campo_edit")
            campo = bloco[campo_opc]
            markers = list(campo.keys())
            marker_edit = st.selectbox("Escolha o marcador:", markers, key="marker_edit")
            current_vars = campo[marker_edit]["vars"]
            st.write(f"Vars atuais: {current_vars}")
            new_vars_str = st.text_input("Digite os novos vars separados por vírgula (ex: 0.1,0.2):", key="new_vars_edit")
            if st.button("Atualizar Vars"):
                new_vars = [v.strip() for v in new_vars_str.split(",") if v.strip()]
                new_vars = sorted(list(set(new_vars)))  # Remover duplicatas e ordenar
                if not new_vars:
                    st.error("❌ Vars inválidos.")
                    return
                campo[marker_edit]["vars"] = new_vars
                salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                st.success(f"✅ Vars atualizados para {marker_edit}: {new_vars}")

            # Gerar vars automaticamente com dicionário de sinônimos
            token = campo[marker_edit]["token"]
            word_to_search = new_vars_str.strip().split(',')[0].strip() if new_vars_str.strip() else token
            if st.button("Gerar Vars com Dicionário", key="gerar_vars_dict"):
                try:
                    import re
                    import unidecode
                    st.write(f"Buscando sinônimos para a palavra: '{word_to_search}'")
                    clean_token = unidecode.unidecode(word_to_search.lower())
                    url = f"https://www.sinonimos.com.br/{clean_token}"
                    st.write(f"URL consultada: {url}")
                    content = ""
                    try:
                        # Tentar com requests primeiro
                        import requests
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                        response = requests.get(url, headers=headers)
                        st.write(f"Status da resposta (requests): {response.status_code}")
                        if response.status_code == 200:
                            content = response.text
                    except ImportError:
                        st.warning("Requests não disponível, tentando Selenium...")
                    
                    if not content:
                        # Fallback para Selenium
                        from selenium import webdriver
                        from selenium.webdriver.chrome.options import Options
                        options = Options()
                        options.add_argument("--headless")
                        options.add_argument("--no-sandbox")
                        options.add_argument("--disable-dev-shm-usage")
                        driver = webdriver.Chrome(options=options)
                        driver.get(url)
                        content = driver.page_source
                        driver.quit()
                        st.write("Conteúdo obtido via Selenium.")
                    
                    candidates = []
                    if content:
                        syn_links = re.findall(r'<a href="https://www\.sinonimos\.com\.br/[^"]+">([^<]+)</a>', content)
                        candidates = [s for s in syn_links if s.lower() != word_to_search.lower() and len(s) > 1][:5]
                    
                    if not candidates:
                        # Tentar com Selenium se requests não encontrou
                        st.write("Tentando com Selenium...")
                        try:
                            from selenium import webdriver
                            from selenium.webdriver.chrome.options import Options
                            options = Options()
                            options.add_argument("--headless")
                            options.add_argument("--no-sandbox")
                            options.add_argument("--disable-dev-shm-usage")
                            driver = webdriver.Chrome(options=options)
                            driver.get(url)
                            content = driver.page_source
                            driver.quit()
                            st.write("Conteúdo obtido via Selenium.")
                            syn_links = re.findall(r'<a href="https://www\.sinonimos\.com\.br/[^"]+">([^<]+)</a>', content)
                            candidates = [s for s in syn_links if s.lower() != word_to_search.lower() and len(s) > 1][:5]
                        except Exception as e:
                            st.error(f"❌ Erro com Selenium: {e}")
                    
                    if candidates:
                        st.write(f"Sugestões geradas: {candidates}")
                        selected = st.multiselect("Selecione as vars para adicionar:", candidates, key=f"select_{marker_edit}")
                        if st.button("Adicionar Selecionadas", key=f"add_{marker_edit}"):
                            current_vars = campo[marker_edit]["vars"]
                            new_vars = list(set(current_vars + selected))  # Evitar duplicatas
                            campo[marker_edit]["vars"] = new_vars
                            salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)
                            st.success(f"✅ Vars adicionadas: {selected}")
                    else:
                        st.warning("⚠️ Nenhuma variação válida encontrada.")
                except ImportError as e:
                    if 'unidecode' in str(e):
                        st.error("❌ Biblioteca 'unidecode' não instalada. Instale com: pip install unidecode")
                    elif 'selenium' in str(e):
                        st.error("❌ Biblioteca 'selenium' não instalada. Instale com: pip install selenium")
                    else:
                        st.error(f"❌ Erro de import: {e}")
                except Exception as e:
                    st.error(f"❌ Erro ao buscar: {e}")
    elif sub_opc == "✏️ Editar nomes de IMs":
        ims = list(memoria.get("IM", {}).keys())
        if not ims:
            st.info("Nenhum IM encontrado.")
            return
        st.write("Edite os nomes dos IMs:")
        for im_id in ims:
            current_name = memoria["IM"][im_id].get("nome", f"IM_{im_id}")
            new_name = st.text_input(f"Nome do IM {im_id}:", value=current_name, key=f"name_{im_id}")
            if st.button(f"Salvar nome para IM {im_id}", key=f"save_name_{im_id}"):
                memoria["IM"][im_id]["nome"] = new_name
                salvar_json(ARQUIVO_MEMORIA, memoria)
                st.success(f"Nome do IM {im_id} atualizado para '{new_name}'!")
                st.rerun()
            
    elif sub_opc == "💾 Backup JSON":
        submenu_backup(memoria, inconsciente)
    
    elif sub_opc == "⬅️ Voltar ao menu principal":
        st.session_state.menu = "principal"


def generate_block_from_template(memoria: dict, template: str) -> None:
    # Parsing mais robusto para textos grandes
    im_id = None
    entrada_texto = ""
    entrada_reacao = ""
    entrada_contexto = ""
    entrada_pensamento = ""
    saidas_textos = []
    saida_reacao = ""
    saida_contexto = ""
    
    # Encontrar seções
    entrada_start = template.find("Entrada:")
    saida_start = template.find("Saída:")
    
    if entrada_start == -1 or saida_start == -1:
        raise ValueError("Template inválido: seções 'Entrada:' e 'Saída:' são obrigatórias")
    
    # Extrair IM ID se presente
    lines = template[:entrada_start].split('\n')
    for line in lines:
        if line.startswith("Índice mãe:"):
            im_id = line.split(":", 1)[1].strip()
            break
    
    # Extrair entrada
    entrada_section = template[entrada_start:saida_start].strip()
    entrada_lines = entrada_section.split('\n')
    
    # Se a primeira linha começa com "Entrada:", capturar o texto
    if entrada_lines and entrada_lines[0].startswith("Entrada:"):
        entrada_texto = entrada_lines[0].split(":", 1)[1].strip()
        entrada_lines = entrada_lines[1:]
    
    current_field = None
    for line in entrada_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Reação:"):
            entrada_reacao = line.split(":", 1)[1].strip()
            current_field = "reacao"
        elif line.startswith("Contexto:"):
            entrada_contexto = line.split(":", 1)[1].strip()
            current_field = "contexto"
        elif line.startswith("Pensamento Interno:"):
            entrada_pensamento = line.split(":", 1)[1].strip()
            current_field = "pensamento"
        elif current_field == "reacao":
            entrada_reacao += " " + line
        elif current_field == "contexto":
            entrada_contexto += " " + line
        elif current_field == "pensamento":
            entrada_pensamento += " " + line
        else:
            # Assume que é continuação do texto de entrada
            if entrada_texto:
                entrada_texto += " " + line
            else:
                entrada_texto = line
    
    # Extrair saída
    saida_section = template[saida_start:].strip()
    saida_lines = saida_section.split('\n')[1:]  # Pular "Saída:"
    
    current_field = None
    for line in saida_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Reação:"):
            saida_reacao = line.split(":", 1)[1].strip()
            current_field = "reacao"
        elif line.startswith("Contexto:"):
            saida_contexto = line.split(":", 1)[1].strip()
            current_field = "contexto"
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
            texto = line.split(".", 1)[1].strip()
            saidas_textos.append(texto)
            current_field = "texto"
        elif current_field == "reacao":
            saida_reacao += " " + line
        elif current_field == "contexto":
            saida_contexto += " " + line
        elif current_field == "texto":
            saidas_textos[-1] += " " + line
    
    if not im_id or not entrada_texto or not saidas_textos:
        raise ValueError("Template inválido: IM ID, texto de entrada e textos de saída são obrigatórios")
    
    # Resto do código permanece o mesmo
    if im_id not in memoria["IM"]:
        memoria["IM"][im_id] = {"nome": f"IM_{im_id}", "genero": "não binário", "ultimo_child": f"{im_id}.0", "blocos": []}
    universo = memoria["IM"][im_id]
    blocos = universo["blocos"]
    next_id = len(blocos) + 1
    bloco = {
        "bloco_id": next_id,
        "entrada": {
            "texto": entrada_texto,
            "reacao": entrada_reacao,
            "contexto": entrada_contexto,
            "pensamento_interno": entrada_pensamento,
            "tokens": {},
            "fim": "",
            "alnulu": len(entrada_texto)
        },
        "saidas": [{
            "textos": saidas_textos,
            "reacao": saida_reacao,
            "contexto": saida_contexto,
            "tokens": {},
            "fim": ""
        }],
        "open": True
    }
    blocos.append(bloco)
    current_last = universo["ultimo_child"]
    E = Token(entrada_texto)
    RE = [entrada_reacao] if entrada_reacao else []
    CE = Token(entrada_contexto)
    pensamento_limpo = entrada_pensamento.strip('"')
    partes = pensamento_limpo.split('.')[:3]
    PIDE_full = []
    for parte in partes:
        PIDE_full.extend(Token(parte.strip()))
    PIDE_limited = PIDE_full[:3]
    S = []
    for t in saidas_textos:
        S += Token(t)
    RS = [saida_reacao] if saida_reacao else []
    CS = Token(saida_contexto)
    entrada_tokens = E + RE + CE + PIDE_full
    saida_tokens = S + RS + CS
    ent_marks_inco = generate_markers(current_last, len(entrada_tokens))
    out_marks = generate_markers(ent_marks_inco[-1], len(saida_tokens))
    fim_ent = ent_marks_inco[-1]
    fim_out = out_marks[-1]
    ent_marks = ent_marks_inco[:len(E) + len(RE) + len(CE) + len(PIDE_limited)]
    idx = 0
    E_m = ent_marks[idx: idx + len(E)]; idx += len(E)
    RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
    CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
    PIDE_m = ent_marks[idx: idx + len(PIDE_limited)]
    jdx = 0
    S_m = out_marks[jdx: jdx + len(S)]; jdx += len(S)
    RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
    CS_m = out_marks[jdx: jdx + len(CS)]
    bloco["entrada"]["tokens"] = {
        "E": E_m,
        "RE": RE_m,
        "CE": CE_m,
        "PIDE": PIDE_m,
        "TOTAL": ent_marks_inco
    }
    bloco["entrada"]["fim"] = fim_ent
    bloco["saidas"][0]["tokens"] = {
        "S": S_m,
        "RS": RS_m,
        "CS": CS_m,
        "TOTAL": out_marks
    }
    bloco["saidas"][0]["fim"] = fim_out
    universo["ultimo_child"] = fim_out
    salvar_json(ARQUIVO_MEMORIA, memoria)
    inconsciente = st.session_state.inconsciente
    bloco_data = {
        "Bloco_id": str(next_id),
        "Entrada": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(ent_marks_inco, entrada_tokens)},
        "SAÍDA": {m: {"token": t, "vars": ["0.0"]} for m, t in zip(out_marks, saida_tokens)}
    }
    if im_id in inconsciente.get("INCO", {}):
        inconsciente["INCO"][im_id]["Blocos"].append(bloco_data)
        inconsciente["INCO"][im_id]["Ultimo child"] = fim_out
    else:
        inconsciente["INCO"][im_id] = {
            "NOME": universo["nome"],
            "Ultimo child": fim_out,
            "Blocos": [bloco_data]
        }
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def recalcular_marcadores_im(memoria: dict, im_id: str) -> None:
    """Recalcula marcadores e tokens para todos os blocos do IM após edição."""
    universo = memoria["IM"][im_id]
    blocos = universo["blocos"]
    if not blocos:
        universo["ultimo_child"] = f"{im_id}.0"
        salvar_json(ARQUIVO_MEMORIA, memoria)
        return

    # Ordenar blocos por id
    blocos.sort(key=lambda b: b["bloco_id"])
    current_last = f"{im_id}.0"

    for bloco in blocos:
        # Retokenizar entrada
        E = Token(bloco["entrada"]["texto"])
        RE = [bloco["entrada"]["reacao"]] if bloco["entrada"]["reacao"] else []
        CE = Token(bloco["entrada"]["contexto"])
        pensamento_limpo = bloco["entrada"]["pensamento_interno"].strip('"')
        partes = pensamento_limpo.split('.')[:3]
        PIDE_full = []
        for parte in partes:
            PIDE_full.extend(Token(parte.strip()))
        PIDE_limited = PIDE_full[:3]

        # Saída
        S = []
        for t in bloco["saidas"][0]["textos"]:
            S += Token(t)
        RS = [bloco["saidas"][0]["reacao"]] if bloco["saidas"][0]["reacao"] else []
        CS = Token(bloco["saidas"][0]["contexto"])

        # Calcular tokens completos
        entrada_tokens = E + RE + CE + PIDE_full
        saida_tokens = S + RS + CS

        # Gerar marcadores alinhados sem sobreposição
        ent_marks_inco = generate_markers(current_last, len(entrada_tokens))
        out_marks = generate_markers(ent_marks_inco[-1], len(saida_tokens))

        fim_ent = ent_marks_inco[-1]
        fim_out = out_marks[-1]

        # Para compatibilidade, ent_marks é o limitado
        ent_marks = ent_marks_inco[:len(E) + len(RE) + len(CE) + len(PIDE_limited)]

        # Subdivide
        idx = 0
        E_m = ent_marks[idx: idx + len(E)]; idx += len(E)
        RE_m = ent_marks[idx: idx + len(RE)]; idx += len(RE)
        CE_m = ent_marks[idx: idx + len(CE)]; idx += len(CE)
        PIDE_m = ent_marks[idx: idx + len(PIDE_limited)]

        jdx = 0
        S_m = out_marks[jdx: jdx + len(S)]; jdx += len(S)
        RS_m = out_marks[jdx: jdx + len(RS)]; jdx += len(RS)
        CS_m = out_marks[jdx: jdx + len(CS)]

        # Atualizar bloco existente
        bloco["entrada"]["tokens"] = {
            "E": E_m,
            "RE": RE_m,
            "CE": CE_m,
            "PIDE": PIDE_m,
            "TOTAL": ent_marks_inco
        }
        bloco["entrada"]["fim"] = fim_ent
        bloco["entrada"]["alnulu"] = len(bloco["entrada"]["texto"])

        bloco["saidas"][0]["tokens"] = {
            "S": S_m,
            "RS": RS_m,
            "CS": CS_m,
            "TOTAL": out_marks
        }
        bloco["saidas"][0]["fim"] = fim_out

        current_last = fim_out

    universo["ultimo_child"] = current_last
    salvar_json(ARQUIVO_MEMORIA, memoria)

    # Atualizar inconsciente - recriar baseado nos blocos, preservando vars existentes por token
    inconsciente = st.session_state.inconsciente
    # Carregar vars existentes por token
    existing_vars_by_token = {}
    if im_id in inconsciente.get("INCO", {}):
        for bloco_inco in inconsciente["INCO"][im_id].get("Blocos", []):
            bloco_id = bloco_inco["Bloco_id"]
            existing_vars_by_token[bloco_id] = {
                "Entrada": {data["token"]: data["vars"] for data in bloco_inco["Entrada"].values()},
                "SAÍDA": {data["token"]: data["vars"] for data in bloco_inco["SAÍDA"].values()}
            }
    
    if im_id not in inconsciente.get("INCO", {}):
        inconsciente.setdefault("INCO", {})[im_id] = {
            "NOME": universo["nome"],
            "Ultimo child": universo["ultimo_child"],
            "Blocos": []
        }
    im_data = inconsciente["INCO"][im_id]
    im_data["Ultimo child"] = universo["ultimo_child"]
    im_data["Blocos"] = []
    for bloco in blocos:
        # Retokenizar para obter tokens atuais
        E = Token(bloco["entrada"]["texto"])
        RE = [bloco["entrada"]["reacao"]] if bloco["entrada"]["reacao"] else []
        CE = Token(bloco["entrada"]["contexto"])
        pensamento_limpo = bloco["entrada"]["pensamento_interno"].strip('"')
        partes = pensamento_limpo.split('.')[:3]
        PIDE_full = []
        for parte in partes:
            PIDE_full.extend(Token(parte.strip()))
        
        S = []
        for t in bloco["saidas"][0]["textos"]:
            S += Token(t)
        RS = [bloco["saidas"][0]["reacao"]] if bloco["saidas"][0]["reacao"] else []
        CS = Token(bloco["saidas"][0]["contexto"])
        
        entrada_tokens_list = E + RE + CE + PIDE_full
        saida_tokens_list = S + RS + CS
        
        entrada_tokens = bloco["entrada"]["tokens"]["TOTAL"]  # marcadores
        saida_tokens = bloco["saidas"][0]["tokens"]["TOTAL"]  # marcadores
        
        # Usar vars existentes por token
        bloco_id = str(bloco["bloco_id"])
        entrada_vars_by_token = existing_vars_by_token.get(bloco_id, {}).get("Entrada", {})
        saida_vars_by_token = existing_vars_by_token.get(bloco_id, {}).get("SAÍDA", {})
        
        bloco_data = {
            "Bloco_id": bloco_id,
            "Entrada": {m: {"token": t, "vars": entrada_vars_by_token.get(t, ["0.0"])} for m, t in zip(entrada_tokens, entrada_tokens_list)},
            "SAÍDA": {m: {"token": t, "vars": saida_vars_by_token.get(t, ["0.0"])} for m, t in zip(saida_tokens, saida_tokens_list)}
        }
        im_data["Blocos"].append(bloco_data)
    salvar_json(ARQUIVO_INCONSCIENTE, inconsciente)


def submenu_estatisticas(memoria: dict) -> None:
    st.subheader("📊 Estatísticas do Sistema INSEPA")
    
    # Número de IMs
    num_ims = len(memoria.get("IM", {}))
    st.metric("Número de IMs", num_ims)
    
    if num_ims > 0:
        # Dados para gráficos
        im_names = []
        num_blocos = []
        total_blocos = 0
        for im_id, im_data in memoria["IM"].items():
            nome = im_data.get("nome", f"IM_{im_id}")
            blocos = len(im_data.get("blocos", []))
            im_names.append(nome)
            num_blocos.append(blocos)
            total_blocos += blocos
        
        st.metric("Total de Blocos", total_blocos)
        
        # Gráfico de barras: Blocos por IM
        import pandas as pd
        df_blocos = pd.DataFrame({"IM": im_names, "Blocos": num_blocos})
        st.bar_chart(df_blocos.set_index("IM"))
        
        # Estatísticas adicionais
        if total_blocos > 0:
            avg_blocos = total_blocos / num_ims
            st.metric("Média de Blocos por IM", f"{avg_blocos:.1f}")
            
            # Distribuição de reações
            reacoes = {}
            for im_data in memoria["IM"].values():
                for bloco in im_data.get("blocos", []):
                    reac = bloco["entrada"].get("reacao", "")
                    if reac:
                        reacoes[reac] = reacoes.get(reac, 0) + 1
            
            if reacoes:
                df_reacoes = pd.DataFrame(list(reacoes.items()), columns=["Reação", "Contagem"])
                st.subheader("Distribuição de Reações de Entrada")
                st.bar_chart(df_reacoes.set_index("Reação"))
        
        # Verificar se há checkpoints treinados
        import os
        ckpts = [f for f in os.listdir(".") if f.startswith("insepa_") and f.endswith(".pt")]
        st.metric("Modelos Treinados", len(ckpts))
        if ckpts:
            st.write("Modelos disponíveis:")
            for ckpt in ckpts:
                dom = ckpt.replace("insepa_", "").replace(".pt", "")
                st.write(f"- {dom}")
    else:
        st.info("Nenhum IM criado ainda.")


def submenu_backup(memoria: dict, inconsciente: dict) -> None:
    st.subheader("💾 Backup dos JSONs")
    st.write("Aqui você pode visualizar e baixar cópias dos JSONs de memória e inconsciente.")
    
    # Backup da Memória
    st.subheader("📄 JSON de Memória (Adam_Lovely_memory.json)")
    memoria_json = json.dumps(memoria, ensure_ascii=False, indent=2)
    st.code(memoria_json, language="json")
    st.download_button(
        label="📥 Baixar JSON de Memória",
        data=memoria_json,
        file_name="Adam_Lovely_memory.json",
        mime="application/json",
        key="download_memoria"
    )
    
    # Backup do Inconsciente
    st.subheader("🧠 JSON do Inconsciente (Adam_Lovely_inconscious.json)")
    inconsciente_json = json.dumps(inconsciente, ensure_ascii=False, indent=2)
    st.code(inconsciente_json, language="json")
    st.download_button(
        label="📥 Baixar JSON do Inconsciente",
        data=inconsciente_json,
        file_name="Adam_Lovely_inconscious.json",
        mime="application/json",
        key="download_inconsciente"
    )
    
    st.info("💡 Use esses backups para restaurar dados ou para deploy. Os arquivos são salvos com timestamp após treinamentos automáticos.")
    
    # Opções de Restauração e Manutenção
    st.subheader("🔧 Restauração e Manutenção")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Limpar Cache Streamlit"):
            # Limpar caches do Streamlit
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache do Streamlit limpo!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reiniciar Sessão"):
            # Limpar session_state e recarregar
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Sessão reiniciada! Recarregando...")
            st.rerun()
    
    with col3:
        if st.button("💾 Fazer Backup Manual"):
            # Salvar backups manuais com timestamp
            import time
            timestamp = int(time.time())
            backup_memoria_file = f"Adam_Lovely_memory_manual_backup_{timestamp}.json"
            backup_inconsciente_file = f"Adam_Lovely_inconscious_manual_backup_{timestamp}.json"
            salvar_json(backup_memoria_file, memoria)
            salvar_json(backup_inconsciente_file, inconsciente)
            st.success(f"✅ Backups manuais salvos: {backup_memoria_file} e {backup_inconsciente_file}")
    
    st.warning("⚠️ **Atenção:** 'Reiniciar Sessão' limpa todos os dados não salvos. Faça backup antes!")


def featurize(field: str, bloco: dict, max_len: int, vocab: dict, val_to_idx: dict, max_ng: int):
    tokens = bloco["entrada"]["tokens"].get(field, [])
    ngrams_list = [generate_ngrams(t, N_GRAM) for t in tokens]
    ids = [vocab.get(ng, vocab.get(UNK, 0)) for nglist in ngrams_list for ng in nglist]
    val_idxs = [val_to_idx.get(t, 0) for t in tokens]
    vals = [float(t) for t in tokens]
    moms = [int(t.split(".", 1)[0]) for t in tokens]
    if vals:
        min_v, max_v = min(vals), max(vals)
        pos = [(v - min_v) / (max_v - min_v) if max_v > min_v else 0.0 for v in vals]
    else:
        pos = []
    pad_ids = (max_len * max_ng) - len(ids)
    pad_vals = max_len - len(tokens)
    ids += [0] * pad_ids
    val_idxs += [0] * pad_vals
    vals += [0.0] * pad_vals
    moms += [0] * pad_vals
    pos += [0.0] * pad_vals
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([val_idxs], dtype=torch.long),
        torch.tensor([vals], dtype=torch.float32),
        torch.tensor([moms], dtype=torch.long),
        torch.tensor([pos], dtype=torch.float32),
    )


def get_feature_vector(bloco: dict, model_params: tuple):
    (state,
     maxE, maxRE, maxCE, maxPIDE,
     mom_size, val_to_idx_E, val_to_idx_RE, val_to_idx_CE, val_to_idx_PIDE,
     vE, vRE, vCE, vPIDE,
     n_txt, n_emo, n_ctx,
     max_ng
     ) = model_params

    E_ids, E_val_idxs, E_val, E_mom, E_pos = featurize("E", bloco, maxE, vE, val_to_idx_E, max_ng)
    RE_ids, RE_val_idxs, RE_val, RE_mom, RE_pos = featurize("RE", bloco, maxRE, vRE, val_to_idx_RE, max_ng)
    CE_ids, CE_val_idxs, CE_val, CE_mom, CE_pos = featurize("CE", bloco, maxCE, vCE, val_to_idx_CE, max_ng)
    PI_ids, PI_val_idxs, PI_val, PI_mom, PI_pos = featurize("PIDE", bloco, maxPIDE, vPIDE, val_to_idx_PIDE, max_ng)

    # Concatenar todos os tensores em um vetor de features
    features = torch.cat([
        E_ids.flatten(), E_val_idxs.flatten(), E_val.flatten(), E_mom.flatten(), E_pos.flatten(),
        RE_ids.flatten(), RE_val_idxs.flatten(), RE_val.flatten(), RE_mom.flatten(), RE_pos.flatten(),
        CE_ids.flatten(), CE_val_idxs.flatten(), CE_val.flatten(), CE_mom.flatten(), CE_pos.flatten(),
        PI_ids.flatten(), PI_val_idxs.flatten(), PI_val.flatten(), PI_mom.flatten(), PI_pos.flatten(),
    ], dim=0)

    return features


def find_similar_blocks(bloco, memoria, dominio, model_params=None):
    blocos = memoria["IM"][dominio]["blocos"]
    if model_params is None:
        # Fallback para o método antigo se não houver model_params
        similar = []
        current_contexto = bloco["entrada"]["contexto"]
        current_reacao = bloco["entrada"]["reacao"]
        for b in blocos:
            if b["bloco_id"] == bloco["bloco_id"]:
                continue
            if b["entrada"]["contexto"] == current_contexto or b["entrada"]["reacao"] == current_reacao:
                similar.append(b)
        return similar[:3]

    # Usar similaridade de cosseno nos vetores de features
    import torch.nn.functional as F
    current_features = get_feature_vector(bloco, model_params)
    similarities = []
    for b in blocos:
        if b["bloco_id"] == bloco["bloco_id"]:
            continue
        b_features = get_feature_vector(b, model_params)
        sim = F.cosine_similarity(current_features.unsqueeze(0), b_features.unsqueeze(0), dim=1).item()
        similarities.append((b, sim))
    # Ordenar por similaridade decrescente e pegar top 3 com sim > 0.7
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar = [b for b, sim in similarities if sim > 0.7][:3]
    return similar


def generate_variation_for_block(bloco, dominio):
    texts = bloco["saidas"][0]["textos"]
    emoji = bloco["saidas"][0].get("reacao", "")
    all_variations = []
    for txt in texts:
        variations = [txt]
        bloco_inco = next((b for b in st.session_state.inconsciente["INCO"][dominio]["Blocos"] if b["Bloco_id"] == str(bloco["bloco_id"])), None)
        if bloco_inco:
            for marker, data in bloco_inco["SAÍDA"].items():
                if data["token"] in txt:
                    valid_vars = [v for v in data["vars"] if v != "0.0"]
                    for var in valid_vars:
                        varied_txt = txt.replace(data["token"], var)
                        variations.append(varied_txt)
        all_variations.extend(variations)
    multivars_saida = bloco["saidas"][0].get("Multivars_Saída", [])
    all_variations.extend(multivars_saida)
    bloco_id = str(bloco["bloco_id"])
    chosen = weighted_choice(all_variations, bloco_id)
    chosen = variar_texto(chosen, bloco, dominio)
    return f"{chosen} {emoji}"


def add_new_block_from_similar(memoria, dominio, sim_bloco, variation):
    universo = memoria["IM"][dominio]
    blocos = universo["blocos"]
    next_id = len(blocos) + 1
    new_bloco = {
        "bloco_id": next_id,
        "entrada": sim_bloco["entrada"].copy(),
        "saidas": [{
            "textos": [variation.split()[0] if variation else "Resposta"],
            "reacao": variation.split()[-1] if len(variation.split()) > 1 else "",
            "contexto": sim_bloco["saidas"][0]["contexto"],
            "tokens": {},
            "fim": ""
        }],
        "open": True
    }
    blocos.append(new_bloco)
    recalcular_marcadores_im(memoria, dominio)
    salvar_json(ARQUIVO_MEMORIA, memoria)
    salvar_json(ARQUIVO_INCONSCIENTE, st.session_state.inconsciente)


def main():
    st.set_page_config(layout="wide")
    # Para deploy no Streamlit Cloud ou similar:
    # 1. Faça upload do código para um repositório Git (GitHub).
    # 2. Vá para share.streamlit.io, conecte o repo e deploy.
    # 3. Para persistência, os dados ficam em session_state; arquivos JSON são backups locais.
    # Nota: Treinamento de IA pode ser lento na nuvem gratuita; considere recursos pagos se necessário.
    # CSS harmonizado: fundo roxo, menu preto com títulos brancos
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 50'%3E%3Ctext fill='rgba(255,255,255,0.1)' font-size='20' x='150' y='25' text-anchor='middle'%3E🤖 Adam 😊 Amor 💜 INSEPA 🌟%3C/text%3E%3C/svg%3E");
        background-repeat: repeat;
        background-size: 300px 50px;
    }
    .stButton>button {
        background: black;
        color: white;
        border: 1px solid white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background: #333;
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
    }
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid white;
    }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
    }
    .stSidebar {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 Adam Lovely AI - Sistema INSEPA")
    st.markdown("### Interface de Chat com IA Avançada")
    # Inicializar dados em session_state para persistência na nuvem
    # Sempre recarregar dados do arquivo para garantir sincronização
    st.session_state.memoria = carregar_json(ARQUIVO_MEMORIA, {"IM": {}})
    st.session_state.inconsciente = carregar_json(ARQUIVO_INCONSCIENTE, {"INCO": {}})
    if "likes" not in st.session_state:
        st.session_state.likes = {}  # {bloco_id: {variacao: count}}
    memoria = st.session_state.memoria
    inconsciente = st.session_state.inconsciente

    # Menu no canto esquerdo
    with st.sidebar:
        st.header("Menu")
        if st.button("🏗️ Gerenciar IMs"):
            st.session_state.menu = "gerenciar"
        if st.button("🧠 Treinar"):
            st.session_state.menu = "treinar"
        if st.button("🧪 Testar"):
            st.session_state.menu = "testar"
        if st.button("💬 Conversar"):
            st.session_state.menu = "conversar"
        if st.button("📊 Estatísticas"):
            st.session_state.menu = "estatisticas"
        if st.button("❌ Sair"):
            st.write("👋 Até mais!")
            st.stop()

        # Modo Administrador
        with st.expander("🔐 Modo Administrador"):
            senha_input = st.text_input("Digite a senha:", type="password", key="admin_senha")
            if st.button("Entrar"):
                if senha_input == SENHA_ADMIN:
                    st.session_state.admin = True
                    st.success("✅ Acesso administrativo concedido!")
                else:
                    st.error("❌ Senha incorreta.")

    if "menu" not in st.session_state:
        st.session_state.menu = "conversar"

    if st.session_state.menu == "gerenciar":
        if not st.session_state.get("admin", False):
            st.error("❌ Acesso negado. Use 'Modo Administrador' no menu lateral para acessar o Gerenciador de IMs.")
            return
        submenu_im(memoria, inconsciente)
    elif st.session_state.menu == "treinar":
        dom = prompt_dominio("treinar", memoria)
        if dom:
            if dom in memoria["IM"]:
                train(memoria, dom)
            else:
                st.error(f"❌ Domínio '{dom}' não encontrado.")
    elif st.session_state.menu == "testar":
        dom = prompt_dominio("testar", memoria)
        if dom:
            if dom in memoria["IM"]:
                test_model(memoria, dom)
            else:
                st.error(f"❌ Domínio '{dom}' não encontrado.")
    elif st.session_state.menu == "conversar":
        st.write("Áudio disponível. Ouça a voz do personagem escolhido agora!")
        dom = prompt_dominio("conversar", memoria)
        if dom:
            if dom in memoria["IM"]:
                infer(memoria, dom)
            else:
                st.error(f"❌ Domínio '{dom}' não encontrado.")
    elif st.session_state.menu == "estatisticas":
        submenu_estatisticas(memoria)


if __name__ == "__main__":
    main()
