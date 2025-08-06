from fastapi import FastAPI
from fastf1 import get_session, Cache
from fastf1.core import Laps
from fastapi.middleware.cors import CORSMiddleware

Cache.enable_cache("cache")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/posicoes/{ano}/{ronda}")
def obter_variacao_posicoes(ano: int, ronda: int):
    session = get_session(ano, ronda, 'R')
    session.load()

    resultados = session.results
    dados = []

    for _, piloto in resultados.iterrows():
        nome = piloto.FullName
        equipa = piloto.TeamName
        grid = piloto.GridPosition
        final = piloto.Position
        ganho = grid - final
        dados.append({
            "piloto": nome,
            "equipa": equipa,
            "inicio": int(grid),
            "fim": int(final),
            "ganho_perda": int(ganho)
        })

@app.get("/construtores/{ano}/{ronda}")
def obter_classificacao_construtores(ano: int, ronda: int):
    total_pontos = {}

    for i in range(1, ronda + 1):
        session = get_session(ano, i, 'R')
        session.load()
        resultados = session.results

        for _, piloto in resultados.iterrows():
            equipa = piloto.TeamName
            pontos = piloto.Points
            total_pontos[equipa] = total_pontos.get(equipa, 0) + pontos

    return dict(sorted(total_pontos.items(), key=lambda item: item[1], reverse=True))

    return dados
