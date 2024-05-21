import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from prophet.diagnostics import cross_validation

import requests
from bs4 import BeautifulSoup

# URL do site IPEADATA
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

# Faz uma requisição GET ao site e captura a resposta
response = requests.get(url)

# Verifica se a requisição foi bem sucedida
if response.status_code == 200:
    # Cria um objeto BeautifulSoup para analisar o HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Procura pela tabela no HTML analisado (o id ou classe pode variar)
    # Você precisaria inspecionar o HTML para obter o seletor correto
    table = soup.find('table', {'id': 'grd_DXMainTable'})

    # Usa o pandas para ler a tabela HTML diretamente para um DataFrame
    df = pd.read_html(str(table),skiprows=0)[0]

    df.columns = df.iloc[0]
    df = df.drop(0)

    # Mostra as primeiras linhas do DataFrame
    df.head()
else:
    print('Falha ao acessar a página: Status Code', response.status_code)

df['Preço - petróleo bruto - Brent (FOB)'] = df['Preço - petróleo bruto - Brent (FOB)'].astype(float)
df['Preço - petróleo bruto - Brent (FOB)'] = df['Preço - petróleo bruto - Brent (FOB)']/100
df.head()

# Definição das funções para cada guia do aplicativo
def tab1():
    st.title('Tech Challenge')
    st.header('> **O Desafio**\n')
    st.markdown('Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo Brent, que pode ser encontrado no site do Ipea.\n\n'
                'Essa base de dados histórica envolve duas colunas: data e preço (em dólares).\n\n'
                'Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo e que gere insights relevantes para tomada de decisão. “Oh!” é esse Streamlit!\n\n'
                'Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.\n\n'
                'A plataforma está dividida em três abas, acessíveis à esquerda. A aba inicial, chamada \'O Desafio\', fornece uma visão geral dos desafios a serem entregues. Em seguida, temos \'O Negócio\', onde são abordadas informações específicas sobre o petróleo Brent. Por fim, na aba \'O Projeto\', é possível encontrar previsões e análises futuras.\n\n'
                '[Clique aqui para acessar os dados do IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)\n\n'
                '[Clique aqui e acesse todo o conteúdo do trabalho no GitHub](https://github.com/WRogoski/tech_4)')

def tab2():
    st.title('Fatores de influência no preço')
    st.markdown('O preço do petróleo é volátil, sendo que seu preço é regulado através da oferta e demanda. Os principais pontos a serem observados para o acompanhamento do preço são:')
    
    st.markdown('<h5> <b>Decisões da OPEP (Organização dos Países Exportadores de Petróleo):</b></h5>', unsafe_allow_html=True)
    st.markdown('Uma organização intragovernamental, composta por 13 países, que busca principalmente a estabilização dos mercados de petróleo, regulamentação da produção e coordenação de políticas petrolíferas.')

    st.markdown('<h5> <b> Questões geopolíticas:</b></h5>', unsafe_allow_html=True)
    st.markdown('Início ou final de conflitos e tensões em regiões produtoras de petróleo, bem como sansões e políticas internacionais a essas, interferem diretamente na disponibilidade do petróleo global.')

    st.markdown('<h5> <b> Transição energética e Políticas ambientais:</b></h5>', unsafe_allow_html=True)
    st.markdown('Os crescentes investimentos em energias renováveis, bem como o surgimento de novas tecnologias e combustíveis que possam substituir o petróleo e reduzir a emissão de carbono influenciam as expectativas de longo prazo para a demanda.')

    st.markdown('<h5> <b> Economia global:</b></h5>', unsafe_allow_html=True)
    st.markdown('A desaceleração da economia global, algo visto com a pandemia de COVID-19 e os lockdowns, gera uma redução nas atividades industriais, levando a um excesso de oferta e quedas significativas dos preços. Já a retomada econômica, como quando da recuperação econômica pós-pandemia, o efeito é contrário, aumentando a demanda por energia e elevando os preços do barril.')

    st.markdown('<h5> <b> Visualizando as oscilações de preço entre 2020 e 2024</b></h5>', unsafe_allow_html=True)

    # Incorporando o dashboard do Power BI usando um iframe
    st.markdown('### Dashboard do Power BI')
    st.write("Aqui está o dashboard do Power BI incorporado no Streamlit:")
    st.write("Caso prefira, pode acessar o dashboard diretamente no [link](https://app.powerbi.com/view?r=eyJrIjoiZmQyZWU0NDQtMmE3NC00MTg0LTg3NDgtY2E0ZGQxNTBiNDkzIiwidCI6ImUyOTY3ODVjLTEwMTYtNDUxYy1hZjA2LWMwZmQ1M2UxNDYyMyJ9)")
    
    # URL do dashboard do Power BI - link de incorporação
    Petroleo_Brent_FOB = "https://app.powerbi.com/view?r=eyJrIjoiZmQyZWU0NDQtMmE3NC00MTg0LTg3NDgtY2E0ZGQxNTBiNDkzIiwidCI6ImUyOTY3ODVjLTEwMTYtNDUxYy1hZjA2LWMwZmQ1M2UxNDYyMyJ9"
    
    # Incorporando o dashboard do Power BI usando um iframe
    components.iframe(Petroleo_Brent_FOB, width=800, height=600)

    st.markdown('### Principais fatores que influenciaram no preço de 2020 a 2024')

    st.markdown('### Bear Market - Queda de preços')
    st.markdown('#### Economia global:')
    st.markdown('No início do ano de 2020 o mundo estava entrando em lockdown. O distanciamento social e a paralisação de atividades ao redor do globo trouxeram incertezas, dada a desaceleração econômica e a redução no consumo do petróleo, especialmente nos setores de transporte e aviação.')

    st.markdown('### Bull Market - Alta de preços')
    st.markdown('#### Economia global:')
    st.markdown('Com o avanço da vacinação e retomada da atividade industrial a pleno ao redor do mundo, os prognósticos econômicos se mostraram positivos. Com isso foi gerado um aumento de demanda por combustível e, consequentemente, o movimento de preços foi inverso ao do início do período pandêmico.')

    st.markdown('#### Decisões da OPEP:')
    st.markdown('Após decisões que levaram a derrocada nos preços no início da pandemia, os membros da OPEP e seus aliados implementaram cortes significativos na produção, equilibrando o mercado e mantendo o nível de preço. As reduções de produção foram ajustadas gradualmente à medida que a demanda se recuperada, garantindo uma oferta controlada e elevação de preços.')

    st.markdown('#### Questões geopolíticas:')
    st.markdown('A invasão russa à Ucrânia foi o principal fator de aumento recente ao preço do barril. O ataque criou incertezas significativas no mercado de energia, já que muitas sanções foram impostas à Russia, uma das principais fontes globais de petróleo. Além do conflito do leste europeu, tensões envolvendo os países do Oriente Médio, outra importante fonte do produto, afetaram a estabilidade da oferta e dos preços.')



#Renomear as colunas identificadas para o padrão 'ds' (Datestamp) e 'y' (target)
df.rename(columns={'Data': 'ds', 'Preço - petróleo bruto - Brent (FOB)': 'y'}, inplace=True)

#Ajustar a coluna 'ds' para formato de data
df['ds'] = pd.to_datetime(df['ds'])

#Remover nulos da coluna 'y'
df = df.dropna(subset=['y'])

#Ordenando o df por data
df = df.sort_values(by='ds')

plt.figure(figsize = (20,10))
plt.plot(df['ds'], df['y'], label='Brent_FOB')

plt.legend(loc='best')
plt.show()





#Separando as bases de treino e teste na proporção 80% 20%

train_data = df.sample(frac=0.8, random_state=20)
test_data = df.drop(train_data.index)
print(f'training data size : {train_data.shape}')
print(f'testing data size : {test_data.shape}')

#Reduzindo tamanho do df para melhorar a visualização e realizar as análises
df_24 = df[df['ds'].dt.year >= 2024]
df_24 = df_24.sort_values(by='ds')
df_24.head()

#Vizualizando o comportamento do pertóleo no último ano
plt.figure(figsize = (20,10))
plt.plot(df_24['ds'], df_24['y'], label='Brent_FOB')
plt.title('Brent_FOB_avg')
plt.legend(loc='best')
plt.show()


#Criando uma cópia do df já limpo e ajustado para utilizar na análise, mantendo o original salvo para uso posterior
df_2 = df_24

#Definindo a data como índice para que o modelo rode corretamente
df_2 = df_2.set_index('ds')


#Iniciei copiando o df gerado anteriormente para realização da análise com Prophet
df_p2 = df_24

#Confirmando se os comandos foram aceitos corretamente
df_p2.head()

#Separando as bases de treino e teste na proporção 80% 20%

train_data = df_p2.sample(frac=0.80, random_state=20)
test_data = df_p2.drop(train_data.index)
print(f'training data size : {train_data.shape}')
print(f'testing data size : {test_data.shape}')

#Instanciando o Prophet para utilizar uma sazonalidade diária e gerar previsões

modelo = Prophet(daily_seasonality=True)
modelo.fit(train_data)
dataFramefuture = modelo.make_future_dataframe(periods=10, freq='D')
previsao = modelo.predict(dataFramefuture)
previsao.head()

#Visualizando os dados juntamente com as previsões
modelo.plot(previsao, figsize=(20,6));
plt.plot(test_data['ds'], test_data['y'], '.r')

modelo.plot_components(previsao, figsize=(10,6))

#Extraindo as colunas relevantes dos DataFrames e mesclando os DataFrames nas colunas 'ds' para comparar previsões e valores reais
previsao_cols = ['ds', 'yhat']
valores_reais_cols = ['ds', 'y']

previsao = previsao[previsao_cols]
valores_reais = train_data[valores_reais_cols]

resultados = pd.merge(previsao, valores_reais, on='ds', how='inner')

#Calcular o erro percentual absoluto para cada ponto de dados e o MAPE
resultados['erro_percentual_absoluto'] = np.abs((resultados['y'] - resultados['yhat']) / resultados['y']) * 100

mape = np.mean(resultados['erro_percentual_absoluto'])

print(f"MAPE: {mape:.2f}%")




def tab3():
    st.title('Modelo preditivo')
    st.markdown('<h5> <b>Petróleo Brent (FOB) </b></h5>', unsafe_allow_html=True)
    st.markdown('Série histórica de preços do Petróleo Brent (FOB) em 2024')
    imagem1 = r'dtat_price\brentfob.png'
    st.image(imagem1, caption='Sua imagem PNG', use_column_width=True)

    
    st.markdown('<h5> <b>Modelo Preditivo </b></h5>', unsafe_allow_html=True)
    st.markdown(f'Para construção do modelo foi utilizado o Prophet, que, considerando a base de preços de 2024, entregou a melhor predição, apresentando um MAPE de {mape:.2f}%.')
    imagem2 = r'dtat_price\prophet.png'
    st.image(imagem2, caption='Sua imagem PNG', use_column_width=True)

    st.markdown('<h5> <b>Modelo Preditivo </b></h5>', unsafe_allow_html=True)
    st.markdown(f'Para construção do modelo foi utilizado o Prophet, que, considerando a base de preços de 2024, entregou a melhor predição, apresentando um MAPE de {mape:.2f}%.')

# Aplicativo Streamlit
def main():
    st.sidebar.title('Menu de navegação')
    tab_selected = st.sidebar.radio('Selecione uma guia', ('Desafio', 'Influência no preço', 'Modelo preditivo'))

    if tab_selected == 'Desafio':
        tab1()
    elif tab_selected == 'Influência no preço':
        tab2()
    elif tab_selected == 'Modelo preditivo':
        tab3()

if __name__ == "__main__":
    main()
