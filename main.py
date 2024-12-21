import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def carregar_arquivo():
    arquivo = 'dados.xlsx'
    if os.path.exists(arquivo):
        print(f"Carregando arquivo '{arquivo}'...")
        return pd.read_excel(arquivo)
    else:
        raise FileNotFoundError(f"Arquivo '{arquivo}' não encontrado na pasta raiz. Certifique-se de que ele está no local correto.")

def preparar_base(dados):
    dados['data'] = pd.to_datetime(dados['Data'])
    base_geral = dados.groupby(dados['data'].dt.to_period('M')).agg(
        receita=('Receita de taxa de entrega', 'sum'),
        pedidos=('Pedidos', 'sum')
    ).reset_index()
    base_geral['data'] = base_geral['data'].dt.to_timestamp()
    return base_geral

def ajustar_para_prophet(base, coluna_valor):
    return base.rename(columns={'data': 'ds', coluna_valor: 'y'})[['ds', 'y']]

def adicionar_regressor_evento(dados, ano_evento=2022, mes_evento=3):
    dados['promo_marco'] = dados['ds'].apply(
        lambda x: 1 if x.year == ano_evento and x.month == mes_evento else 0
    )
    return dados

def plotar_previsoes(previsoes, dados, titulo, formato_y):
    plt.figure(figsize=(12, 6))
    historico = previsoes[previsoes['ds'] <= dados['ds'].max()]
    projetado = previsoes[previsoes['ds'] > dados['ds'].max()]

    plt.plot(historico['ds'], historico['yhat'], label='Histórico', color='#B3B3B3', linewidth=2)
    plt.plot(projetado['ds'], projetado['yhat'], label='Projeção', color='#D7A1A5', linestyle='--', linewidth=2)
    plt.fill_between(projetado['ds'], projetado['yhat_lower'], projetado['yhat_upper'], color='#D7A1A5', alpha=0.2, label='Intervalo de Confiança')

    valor_inicial = historico['yhat'].iloc[-1]
    valor_final = projetado['yhat'].iloc[-1]
    crescimento_percentual = ((valor_final - valor_inicial) / valor_inicial) * 100
    deslocamento_texto = abs(valor_final - valor_inicial) * 0.7
    texto_y = valor_final - deslocamento_texto if crescimento_percentual > 0 else valor_final + deslocamento_texto

    plt.annotate(
        f"{crescimento_percentual:.2f}%", 
        xy=(projetado['ds'].iloc[-1], valor_final), 
        xytext=(projetado['ds'].iloc[-1], texto_y), 
        fontsize=12, color='#D7A1A5', weight='semibold', ha='center', 
        arrowprops=dict(arrowstyle="->", color='#D7A1A5', lw=1.5)
    )

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(formato_y))
    plt.grid(axis='y', color='#C8C6C4', alpha=0.3, linestyle='--')
    plt.xlabel("")
    plt.ylabel("")
    plt.title(titulo, fontsize=14, weight='bold', color='#333333')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()

def plotar_previsoes_por_segmento(previsoes, dados, titulo):
    for segmento, previsao in previsoes.items():
        dados_segmento = dados[dados['Segmento'] == segmento]
        plt.figure(figsize=(12, 6))
        historico = previsao[previsao['ds'] <= dados_segmento['Data'].max()]
        projetado = previsao[previsao['ds'] > dados_segmento['Data'].max()]
        plt.plot(historico['ds'], historico['yhat'], label='Histórico', color='#B3B3B3', linewidth=2)
        plt.plot(projetado['ds'], projetado['yhat'], label='Projeção', color='#D7A1A5', linestyle='--', linewidth=2)
        
        valor_inicial = historico['yhat'].iloc[-1]
        valor_final = projetado['yhat'].iloc[-1]
        crescimento_percentual = ((valor_final - valor_inicial) / valor_inicial) * 100
        deslocamento_texto = abs(valor_final - valor_inicial) * 0.9
        texto_y = valor_final - deslocamento_texto if crescimento_percentual > 0 else valor_final + deslocamento_texto

        plt.annotate(
            f"{crescimento_percentual:.2f}%", 
            xy=(projetado['ds'].iloc[-1], valor_final), 
            xytext=(projetado['ds'].iloc[-1], texto_y), 
            fontsize=12, color='#D7A1A5', weight='semibold', ha='center', 
            arrowprops=dict(arrowstyle="->", color='#D7A1A5', lw=1.5)
        )

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        plt.grid(axis='y', color='#C8C6C4', alpha=0.3, linestyle='--')

        def formatar_eixo_y(valor, _):
            if valor >= 1e6:
                return f"R${valor / 1e6:.1f} Mi"
            else:
                return f"R${valor / 1e3:.0f} Mil"

        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(formatar_eixo_y))
        plt.title(f"{titulo} - {segmento}", fontsize=14, weight='bold', color='#333333')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend()
        plt.show()

def formatar_valor_milhoes(valor, _):
    return f"R${valor / 1e6:.0f} Mi"

def formatar_valor_inteiros(valor, _):
    return f"{int(valor):,}".replace(',', '.')


base_bruta = carregar_arquivo()
base_geral = preparar_base(base_bruta)

base_receita = ajustar_para_prophet(base_geral, 'receita')
base_pedidos = ajustar_para_prophet(base_geral, 'pedidos')

base_receita = adicionar_regressor_evento(base_receita)
base_pedidos = adicionar_regressor_evento(base_pedidos)

modelo_receita = Prophet(interval_width=0.7)
modelo_receita.add_regressor('promo_marco')
modelo_receita.fit(base_receita)

modelo_pedidos = Prophet(interval_width=0.7)
modelo_pedidos.add_regressor('promo_marco')
modelo_pedidos.fit(base_pedidos)

futuro_receita = modelo_receita.make_future_dataframe(periods=6, freq='M')
futuro_receita['promo_marco'] = futuro_receita['ds'].apply(lambda x: 1 if x.month == 3 else 0)

futuro_pedidos = modelo_pedidos.make_future_dataframe(periods=6, freq='M')
futuro_pedidos['promo_marco'] = futuro_pedidos['ds'].apply(lambda x: 1 if x.month == 3 else 0)

previsao_receita = modelo_receita.predict(futuro_receita)
previsao_pedidos = modelo_pedidos.predict(futuro_pedidos)

plotar_previsoes(previsao_receita, base_receita, "Projeção de Receita com Sazonalidade de Março", formatar_valor_milhoes)
plotar_previsoes(previsao_pedidos, base_pedidos, "Projeção de Pedidos com Sazonalidade de Março", formatar_valor_inteiros)

segmentos = base_bruta['Segmento'].unique()
previsoes_segmento = {}

for segmento in segmentos:
    dados_segmento = base_bruta[base_bruta['Segmento'] == segmento]
    base_segmento = ajustar_para_prophet(preparar_base(dados_segmento), 'receita')
    modelo_segmento = Prophet(interval_width=0.7)
    modelo_segmento.fit(base_segmento)
    futuro_segmento = modelo_segmento.make_future_dataframe(periods=6, freq='M')
    previsoes_segmento[segmento] = modelo_segmento.predict(futuro_segmento)

plotar_previsoes_por_segmento(previsoes_segmento, base_bruta, "Projeção de Receita por Segmento")
