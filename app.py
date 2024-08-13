import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from io import BytesIO

# Criar guias
titulos_guias = ['Introdução','Tratamento do DataFrame', 'Dashboard', 'Modelo Preditivo']
guia1, guia2, guia3, guia4 = st.tabs(titulos_guias)


with guia1:

    st.title('Passos Mágicos')
  
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
     
    st.subheader('Introdução')

    st.markdown("""
    
    A ONG "Passos Mágicos" é uma organização sem fins lucrativos dedicada a transformar a vida de crianças e jovens em situação de vulnerabilidade social através da educação. 
    Desde a sua fundação, a Passos Mágicos tem trabalhado incansavelmente para oferecer oportunidades educacionais e de desenvolvimento pessoal a estudantes de comunidades carentes, 
    acreditando que a educação é a chave para a mudança social e o empoderamento.
    
    """)
    
    st.subheader('Missão')

    st.markdown('''
    
    A missão da Passos Mágicos é instrumentalizar a educação como ferramenta de transformação, promovendo o desenvolvimento integral de crianças e jovens, 
    ajudando-os a superar desafios e a construir um futuro melhor. Através de programas educacionais, atividades extracurriculares e apoio psicológico, 
    a ONG busca garantir que cada aluno atinja seu pleno potencial.
    
    ''')

    st.subheader('Objetivos')

    st.markdown('''
    
    Desenvolvimento Educacional: Proporcionar uma educação de qualidade que fomente o aprendizado contínuo e o pensamento crítico.
    Apoio Psicológico e Social: Oferecer suporte emocional e social para ajudar os alunos a enfrentarem suas dificuldades pessoais e sociais.
    Preparação para o Futuro: Equipar os alunos com habilidades essenciais para a vida e o mercado de trabalho, promovendo a inclusão social e profissional.
    
    ''')
    
    st.subheader('Programas e Atividades')
    
    st.markdown('''
                
    A ONG oferece uma variedade de programas e atividades, incluindo:

    * Aulas de Reforço Escolar: Para ajudar os alunos a melhorarem seu desempenho acadêmico.
    * Atividades Culturais e Esportivas: Para promover o desenvolvimento físico e artístico.
    * Orientação Profissional: Para preparar os jovens para o mercado de trabalho e incentivar a continuidade dos estudos.
         
    ''')  
    
    st.subheader('Impacto na Comunidade')
    
    st.markdown('''
                
    Desde a sua criação, a Passos Mágicos tem impactado positivamente a vida de inúmeros jovens, proporcionando-lhes as ferramentas necessárias para 
    superar barreiras e alcançar seus sonhos. Através de uma abordagem holística, que integra educação, apoio emocional e desenvolvimento pessoal, 
    a ONG tem visto melhorias significativas nos índices de desempenho educacional dos seus alunos.
    
    A Passos Mágicos é mais do que uma instituição de ensino; é uma comunidade comprometida com a transformação social e o desenvolvimento humano. 
    Com o apoio de voluntários, parceiros e doadores, a ONG continua a expandir seu alcance e a fazer uma diferença significativa na vida de muitas crianças e jovens.
  
    ''')
    
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
    
    st.subheader('Pós Tech - FIAP')
    
    st.markdown('Created by:')       
    st.markdown('Leandro Castro - RM 350680')
    st.markdown('Mateus Correa - RM 351094')
    st.markdown('Tatiane Gandra - RM 352177' )
    
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
    
##############################################    
                            

    # Carregar os dados
    df = pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')
    

    with guia2:
        st.title('Tratamento dos Dados')
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
        
        # Exibir os primeiros registros do dataset
        st.markdown("* Exibindo as primeiras linhas do DataFrame para ter uma visão inicial dos dados.")
        st.markdown('')
        st.write(df.head())
        
        ##################################################
        
        # Filtrando colunas por ano
        cols_2020 = [col for col in df.columns if '2020' in col]
        cols_2021 = [col for col in df.columns if '2021' in col]
        cols_2022 = [col for col in df.columns if '2022' in col]

        # Identificar alunos com dados completos em cada ano
        df['has_2020'] = df[cols_2020].notna().any(axis=1)
        df['has_2021'] = df[cols_2021].notna().any(axis=1)
        df['has_2022'] = df[cols_2022].notna().any(axis=1)

        # Alunos com dados em todos os anos
        all_years = df[df['has_2020'] & df['has_2021'] & df['has_2022']]

        # Alunos com dados apenas em 2020
        only_2020 = df[df['has_2020'] & ~df['has_2021'] & ~df['has_2022']]

        # Alunos com dados apenas em 2021
        only_2021 = df[df['has_2021'] & ~df['has_2020'] & ~df['has_2022']]

        # Alunos com dados apenas em 2022
        only_2022 = df[df['has_2022'] & ~df['has_2020'] & ~df['has_2021']]

        # Resultados
        num_all_years = all_years.shape[0]
        num_only_2020 = only_2020.shape[0]
        num_only_2021 = only_2021.shape[0]
        num_only_2022 = only_2022.shape[0]

        st.write(f'* Número de alunos com dados em todos os anos: {num_all_years}')
        st.write(f'* Número de alunos com dados apenas em 2020: {num_only_2020}')
        st.write(f'* Número de alunos com dados apenas em 2021: {num_only_2021}')
        st.write(f'* Número de alunos com dados apenas em 2022: {num_only_2022}')
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        # Exibir as tabelas filtradas
        st.subheader("Alunos com dados em todos os anos")
        st.markdown('')
        st.dataframe(all_years)

        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        st.subheader("Alunos com dados apenas em 2020")
        st.markdown('')
        st.dataframe(only_2020)
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        st.subheader("Alunos com dados apenas em 2021")
        st.markdown('')
        st.dataframe(only_2021)
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        st.subheader("Alunos com dados apenas em 2022")
        st.markdown('')
        st.dataframe(only_2022)
             
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
        
        ##############################################
        
        st.subheader("DataFrame - Alunos com dados em todos os anos")
        
        st.markdown("* Para todas as próximas análises iremos utilizar este dataframe que possui dados em todos os anos")
                
        # Verificar valores faltantes
        st.markdown("* Verificando a quantidade de valores faltantes em cada coluna do dataframe all_years")
        st.markdown("")
        st.write(all_years.isnull().sum())
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
        
        # Tratamento de valores faltantes nas colunas numéricas

        
        st.markdown("* Aplicando a correção para garantir apenas uma casa decimal.")
        
        # Aplicando a correção para garantir apenas uma casa decimal
        columns_to_fix = ['INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 
                        'IDA_2022', 'NOTA_PORT_2022', 'NOTA_MAT_2022', 
                        'NOTA_ING_2022', 'IPP_2022', 'IPV_2022']
        
        for column in columns_to_fix:
            all_years[column] = pd.to_numeric(all_years[column], errors='coerce').round(1)
        
        # Preencher valores faltantes em colunas não numéricas com "Desconhecido"
        st.markdown("* Preenchemos valores faltantes em colunas não numéricas com 'Desconhecido'.")
        non_numerical_columns = all_years.select_dtypes(exclude=['float64', 'int64']).columns
        all_years[non_numerical_columns] = all_years[non_numerical_columns].fillna('Desconhecido')
        
        st.markdown('* A coluna numérica de notas de Inglês, possui 151 valores em brancos, decidimos deixar em branco para manter a fidelidade dos dados.')
        

        # Verificar novamente valores faltantes após o tratamento
        st.markdown("* Verificando novamente os valores faltantes após o preenchimento.")
        st.markdown("")
        st.write(all_years.isnull().sum())
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        st.markdown("* Visualização após as correções")
        
        # Verificar novamente o DataFrame após ajustes
        st.markdown("")
        st.write(all_years.head())

        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
        
        st.markdown('Ajuste da Casa Decimal')
        
        st.markdown('''Mesmo após utilizarmos diversas técnicas para o ajuste, algumas colunas continuaram com mais de uma casa decimal. Sendo assim baixamos o aquivo em Excel e percebemos que as colunas que não conseguimos ajustar 
                    são as colunas que estão com "ponto" separando a casa decimal. A partir disso substituimos os "pontos" por "vírgula", convertemos a coluna para numérica e deixamos apenas uma casa decimal.             
                    ''')
        
        st.markdown('* Planilha antes da substituição do ponto por vírgula')
        
        # Função para converter o DataFrame em um arquivo Excel
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            processed_data = output.getvalue()
            return processed_data

        # Gerar o Excel a partir do DataFrame all_years
        excel_data = to_excel(all_years)

        # Botão para download do arquivo Excel
        st.download_button(label="Baixar Dados em Excel",
                        data=excel_data,
                        file_name='dados.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')  

                
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
        
        st.subheader("Carregando a base corrigida")
        
        df_tratado = pd.read_excel("PEDE_PASSOS_DATASET_FIAP.xlsx")   
        
        st.markdown("Aplicando novamente a correção para garantir apenas uma casa decimal")   
        
        # Aplicando a correção para garantir apenas uma casa decimal
        columns_to_fix = ['INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 
                        'IDA_2022', 'NOTA_PORT_2022', 'NOTA_MAT_2022', 
                        'NOTA_ING_2022', 'IPP_2022', 'IPV_2022']
        
        for column in columns_to_fix:
            df_tratado[column] = pd.to_numeric(df_tratado[column], errors='coerce').round(1)
            
                              
        # Exibir os primeiros registros do dataset
        st.markdown("* Tabela final após todas as correções")
        st.markdown('')
        st.write(df_tratado) 
        
        st.markdown('* Planilha final após todos os tratamentos')
        
        # Gerar o Excel a partir do DataFrame df_tratado
        excel_tratado = to_excel(df_tratado)

        # Botão para download do arquivo Excel tratado
        st.download_button(label="Baixar Planilha Corrigida",
                        data=excel_tratado,
                        file_name='PEDE_PASSOS_DATASET_FIAP_Corrigido.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)            
                
        # Descrição estatística do dataset
        st.markdown("* Descrição estatística")
        st.markdown("")
        st.write(df_tratado.describe())
        
        st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

        
       
       
############################################################      
    
        with guia3:
            
            st.title('Análise Exploratória')
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
            
            # Dashboard de Perfil dos Alunos
            st.markdown('##### Dashboard de Perfil dos Alunos')

            # Filtragem por ano
            ano_selecionado = st.selectbox('Selecione o Ano', ['2020', '2021', '2022'])

            # Gráfico de distribuição de idade
            if f'IDADE_ALUNO_{ano_selecionado}' in df_tratado.columns:
                fig_idade = px.histogram(df_tratado, x=f'IDADE_ALUNO_{ano_selecionado}', nbins=10, title=f'Distribuição de Idade dos Alunos em {ano_selecionado}')
                st.plotly_chart(fig_idade)
            else:
                st.write(f"Dados de idade para {ano_selecionado} não disponíveis.")
                
            st.markdown('* Eixo X: Mostra as faixas de idades dos alunos.')
            st.markdown('* Eixo Y: Mostra quantos alunos estão em cada intervalo de idade.')
                
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True) 
            
            ###############################################

                      
            # Carregar os dados tratados
            df_tratados = pd.read_excel('PEDE_PASSOS_DATASET_FIAP.xlsx')

            # Selecionar ano para exibir o gráfico
            anos = ['2020', '2021', '2022']
            ano_selecionado = st.selectbox('Selecione o ano', anos)

            # Filtrar a coluna INDE para o ano selecionado
            coluna_inde = f'INDE_{ano_selecionado}'
            df_filtrado = df_tratados[coluna_inde].dropna()

            # Criar gráfico de distribuição
            fig = px.histogram(df_filtrado, x=coluna_inde, nbins=20, title=f'Distribuição do INDE em {ano_selecionado}')

            # Exibir o gráfico
            st.plotly_chart(fig)
            
            st.markdown("* Eixo X: Representa os valores do índice INDE para o ano que foi selecionado (por exemplo, INDE_2020, INDE_2021, ou INDE_2022).")
            st.markdown("* Eixo Y: Mostra a contagem (frequência) de ocorrências para cada intervalo de valores de INDE.")
              
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)         

            ################################################            
            
          
            # Evolução do INDE (Índice de Desenvolvimento Educacional)
            st.subheader("INDE (Índice de Desenvolvimento Educacional)")
            df_inde = df_tratado[['INDE_2020', 'INDE_2021', 'INDE_2022']]
            df_inde = pd.melt(df_inde, var_name='Ano', value_name='INDE')
            df_inde['Ano'] = df_inde['Ano'].str.extract('(\d+)')
            fig_inde = px.line(df_inde, x='Ano', y='INDE', title='Evolução do INDE (2020-2022)')
            st.plotly_chart(fig_inde)
            
            st.markdown('* Eixo X: Mostra os anos 2020, 2021 e 2022.')
            st.markdown('* Eixo Y: Mostra os valores do INDE, indicando o desempenho educacional dos alunos em cada um desses anos.')
            
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
            

            # Análise dos Indicadores de Desempenho
            st.subheader("Indicadores de Desempenho")
            indicadores = ['IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV']
            df_indicadores = df_tratado[[f'{ind}_2020' for ind in indicadores] + [f'{ind}_2021' for ind in indicadores] + [f'{ind}_2022' for ind in indicadores]]
            df_indicadores = pd.melt(df_indicadores, var_name='Indicador_Ano', value_name='Valor')
            df_indicadores['Indicador'] = df_indicadores['Indicador_Ano'].str.extract('([A-Z]+)')
            df_indicadores['Ano'] = df_indicadores['Indicador_Ano'].str.extract('(\d+)')
            fig_indicadores = px.line(df_indicadores, x='Ano', y='Valor', color='Indicador', title='Evolução dos Indicadores de Desempenho (2020-2022)')
            st.plotly_chart(fig_indicadores)
            
            st.markdown('* Eixo X: Mostra os anos 2020, 2021 e 2022.')
            st.markdown('* Eixo Y: Mostra os valores dos diferentes indicadores de desempenho.')
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

            # Distribuição do Conceito de Pedras
            st.subheader("Conceito de Pedras")
            df_pedra = df_tratado[['PEDRA_2020', 'PEDRA_2021', 'PEDRA_2022']]
            df_pedra = pd.melt(df_pedra, var_name='Ano', value_name='Pedra')
            df_pedra['Ano'] = df_pedra['Ano'].str.extract('(\d+)')
            fig_pedra = px.histogram(df_pedra, x='Pedra', color='Ano', barmode='group', title='Distribuição do Conceito de Pedras (2020-2022)')
            st.plotly_chart(fig_pedra)
            
            st.markdown('* Eixo X: Mostra as diferentes categorias de Pedras.')
            st.markdown('* Eixo Y: Mostra a contagem de cada categoria de Pedra em cada ano.')
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
            
            ########################################

            # Análise de Ponto de Virada
            st.subheader("Ponto de Virada")
            df_ponto_virada = df_tratado[['PONTO_VIRADA_2020', 'PONTO_VIRADA_2021', 'PONTO_VIRADA_2022']]
            df_ponto_virada = pd.melt(df_ponto_virada, var_name='Ano', value_name='Ponto_Virada')
            df_ponto_virada['Ano'] = df_ponto_virada['Ano'].str.extract('(\d+)')
            fig_ponto_virada = px.histogram(df_ponto_virada, x='Ponto_Virada', color='Ano', barmode='group', title='Análise de Ponto de Virada (2020-2022)')
            st.plotly_chart(fig_ponto_virada)
            
            st.markdown('* Eixo X: Mostra as diferentes categorias de Ponto de Virada "Sim" ou "Não")')
            st.markdown('* Eixo Y: Mostra a contagem de cada categoria de Ponto de Virada em cada ano.')
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
            
            #############################################

            # Distribuição de Notas por Disciplina em diferentes anos
            st.subheader("Distribuição de Notas por Disciplina")

               
            fig_port = px.histogram(df_tratado, x='NOTA_PORT_2022', nbins=10, title='Distribuição de Notas de Português em 2022')
            fig_mat = px.histogram(df_tratado, x='NOTA_MAT_2022', nbins=10, title='Distribuição de Notas de Matemática em 2022')
            fig_ing = px.histogram(df_tratado, x='NOTA_ING_2022', nbins=10, title='Distribuição de Notas de Inglês em 2022')

            st.plotly_chart(fig_port)
            st.plotly_chart(fig_mat)
            st.plotly_chart(fig_ing)     
            
            st.markdown('* Eixo X: Mostra as notas dos alunos em intervalos para Português, Matemática ou Inglês em 2022.')
            st.markdown('* Eixo Y: Mostra a contagem de alunos que obtiveram notas dentro de cada intervalo específico')
            
            st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

            ################################################
            # Jornada dos alunos entre as Pedras
            st.subheader('Jornada dos alunos entre as Pedras')
            # Mapeamento das pedras para um valor numérico (para identificar avanços e recuos)
            pedra_mapping = {
                'Topázio': 1,
                'Ametista': 2,
                'Ágata': 3,
                'Quartzo': 4
            }

            # Supondo que as colunas 'PEDRA_2020', 'PEDRA_2021', 'PEDRA_2022' estejam no DataFrame
            df['PEDRA_2020'] = df['PEDRA_2020'].map(pedra_mapping)
            df['PEDRA_2021'] = df['PEDRA_2021'].map(pedra_mapping)
            df['PEDRA_2022'] = df['PEDRA_2022'].map(pedra_mapping)

            # Função para categorizar o avanço, recuo ou estagnação
            def categorize_movement(start, end):
                if pd.isna(start) or pd.isna(end):
                    return 'Sem dados'
                elif end > start:
                    return 'Avanço'
                elif end < start:
                    return 'Recuo'
                else:
                    return 'Estagnação'

            # Aplicar a função para categorizar a mudança entre os anos
            df['Movimento_2020_2021'] = df.apply(lambda row: categorize_movement(row['PEDRA_2020'], row['PEDRA_2021']), axis=1)
            df['Movimento_2021_2022'] = df.apply(lambda row: categorize_movement(row['PEDRA_2021'], row['PEDRA_2022']), axis=1)

            # Filtrar para remover os casos com "Sem dados"
            df_filtered = df[(df['Movimento_2020_2021'] != 'Sem dados') & (df['Movimento_2021_2022'] != 'Sem dados')]

            # Contagem dos movimentos por tipo (em percentual)
            mov_2020_2021 = df_filtered['Movimento_2020_2021'].value_counts(normalize=True).reindex(['Avanço', 'Recuo', 'Estagnação'], fill_value=0) * 100
            mov_2021_2022 = df_filtered['Movimento_2021_2022'].value_counts(normalize=True).reindex(['Avanço', 'Recuo', 'Estagnação'], fill_value=0) * 100

            # Criar gráfico de barras agrupadas
            fig_jornada = go.Figure()

            fig_jornada.add_trace(go.Bar(
                x=mov_2020_2021.index,
                y=mov_2020_2021,
                name='2020-2021',
                text=mov_2020_2021.apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ))

            fig_jornada.add_trace(go.Bar(
                x=mov_2021_2022.index,
                y=mov_2021_2022,
                name='2021-2022',
                text=mov_2021_2022.apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ))

            # Configurar layout do gráfico
            fig_jornada.update_layout(
                barmode='group',  # Definindo para barras agrupadas
                title='Jornada de Avanço e Recuo entre Conceitos de Pedra',
                xaxis=dict(title='Tipo de Movimento'),
                yaxis=dict(title='Proporção de Estudantes (%)'),
                legend_title='Período'
            )

            st.plotly_chart(fig_jornada)

            ########################

            st.subheader('Análises de defasagem entre a fase que o aluno deveria estar e a que ele realmente está')
            # Mapeamento dos níveis de ensino para valores numéricos
            nivel_mapping_2021 = {
                'Nível 1 (1o ao 4o ano)': 1,
                'Nível 2 (5o e 6o ano)': 2,
                'Nível 3 (7o e 8o ano)': 3,
                'Nível 4 (9o ano)': 4,
                'Nível 5 (1o EM)': 5,
                'Nível 6 (2o EM)': 6,
                'Nível 7 (3o EM)': 7
            }

            nivel_mapping_2022 = {
                'Fase 1 (1º ao 4º ano)': 1,
                'Fase 2 (5º e 6º ano)': 2,
                'Fase 3 (7º e 8º ano)': 3,
                'Fase 4 (9º ano)': 4,
                'Fase 5 (1º EM)': 5,
                'Fase 6 (2º EM)': 6,
                'Fase 7 (3º EM)': 7,
                'ALFA  (2º e 3º ano)': 0  # Se precisar de uma categorização especial para ALFA
            }

            # Aplicar o mapeamento para converter os níveis de ensino para valores numéricos
            df['NIVEL_IDEAL_2021'] = df['NIVEL_IDEAL_2021'].map(nivel_mapping_2021)
            df['NIVEL_IDEAL_2022'] = df['NIVEL_IDEAL_2022'].map(nivel_mapping_2022)


            # Calcular a defasagem entre o nível ideal e a fase atual
            df['DEFASAGEM_2021'] = df['NIVEL_IDEAL_2021'] - df['FASE_2021']
            df['DEFASAGEM_2022'] = df['NIVEL_IDEAL_2022'] - df['FASE_2022']


            # Concatenar as defasagens para criar um gráfico de comparação
            defasagem_data = pd.melt(df[['DEFASAGEM_2021', 'DEFASAGEM_2022']], var_name='Ano', value_name='Defasagem')

            # Criar gráfico de violino para mostrar a distribuição da defasagem
            fig_defasagem = px.violin(defasagem_data.dropna(), x='Ano', y='Defasagem', box=True, points="all", title='Distribuição da Defasagem por Ano')

            st.plotly_chart(fig_defasagem)

            ##################################

            # Criar grupos de defasagem para 2021
            df['Grupo_Defasagem_2021'] = pd.cut(df['DEFASAGEM_2021'], bins=[-float('inf'), -0.5, 0.5, float('inf')], labels=['Negativa', 'Neutra', 'Positiva'])

            # Verificar se existe uma coluna relacionada a bolsas em 2021, senão usaremos outra informação
            if 'BOLSISTA_2021' in df.columns:
                bolsa_2021 = 'BOLSISTA_2021'
            elif 'BOLSISTA_2022' in df.columns:  # Caso a coluna exista para 2022
                bolsa_2021 = 'BOLSISTA_2022'
            else:
                bolsa_2021 = None

            # Analisar a distribuição da instituição de ensino para cada grupo em 2021
            inst_ensino_2021 = df.groupby('Grupo_Defasagem_2021')['INSTITUICAO_ENSINO_ALUNO_2021'].value_counts(normalize=True).unstack()

            # Analisar a distribuição de bolsistas para cada grupo em 2021 (se disponível)
            if bolsa_2021:
                bolsa_2021_dist = df.groupby('Grupo_Defasagem_2021')[bolsa_2021].value_counts(normalize=True).unstack()
            else:
                bolsa_2021_dist = "Informação de bolsa não disponível"

            
            # Gráfico 1: Distribuição por Instituição de Ensino em 2021
            fig_escolas = go.Figure()

            # Obtendo os números absolutos para cada grupo e instituição
            totals_ensino = df.groupby(['Movimento_2020_2021', 'INSTITUICAO_ENSINO_ALUNO_2021']).size()

            for coluna in inst_ensino_2021.columns:
                fig_escolas.add_trace(go.Bar(
                    x=inst_ensino_2021.index,
                    y=inst_ensino_2021[coluna] * 100,  # Convertendo para percentual
                    name=coluna,
                    text=inst_ensino_2021[coluna] * 100,  # Texto em percentual
                    texttemplate='%{text:.1f}%',  # Formatando texto como percentual
                    textposition='auto'
                ))

            fig_escolas.update_layout(
                title="Distribuição por Instituição de Ensino em 2021",
                xaxis_title="Grupo de Defasagem",
                yaxis_title="Proporção de Estudantes (%)",
                barmode='group'
            )

            st.plotly_chart(fig_escolas)
                      
            
            ################################################
            
            with guia4:
                
                st.header('Modelo Preditivo')
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                # Subtítulo da aplicação
                st.subheader('Regressão Linear')
                
                st.markdown('')

                # Selecionar features e target para Regressão Linear
                features_2020_2021 = df_tratado[['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020',
                                                'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021']]
                target_2022 = df_tratado['INDE_2022']

                # Configurar e validar o modelo de Regressão Linear
                model_lr = LinearRegression()
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                mse_scores_lr = cross_val_score(model_lr, features_2020_2021, target_2022, cv=kf, scoring='neg_mean_squared_error')
                mse_scores_lr = -mse_scores_lr

                st.write(f'Média do Erro Quadrático Médio (MSE) - Regressão Linear: {np.mean(mse_scores_lr)}')
                st.write(f'Desvio Padrão do MSE - Regressão Linear: {np.std(mse_scores_lr)}')
                
                st.markdown('')

                # Treinar o modelo de Regressão Linear com todos os dados de treino
                model_lr.fit(features_2020_2021, target_2022)

                # Preparar features de 2022 para previsão, mantendo os nomes das colunas de 2020 e 2021
                features_2022 = df_tratado[['INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021',
                                            'INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022']]
                features_2022.columns = ['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020',
                                        'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021']

                # Fazer previsões para 2023 usando os dados de 2022
                predictions_lr = model_lr.predict(features_2022)

                # Criar uma tabela com as previsões
                tabela_previsoes_lr = pd.DataFrame({
                    'NOME': df_tratado['NOME'],
                    'INDE_2022': df_tratado['INDE_2022'],
                    'Previsão_INDE_2023': predictions_lr
                })

                st.write(tabela_previsoes_lr)
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                

                st.subheader('Random Forest')
                
                st.markdown('')
                
                # Configurar e validar o modelo Random Forest
                model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                mse_scores_rf = cross_val_score(model_rf, features_2020_2021, target_2022, cv=kf, scoring='neg_mean_squared_error')
                mse_scores_rf = -mse_scores_rf

                st.write(f'Média do Erro Quadrático Médio (MSE) - Random Forest: {np.mean(mse_scores_rf)}')
                st.write(f'Desvio Padrão do MSE - Random Forest: {np.std(mse_scores_rf)}')
                
                st.markdown('')

                # Treinar o modelo Random Forest com todos os dados de treino
                model_rf.fit(features_2020_2021, target_2022)

                # Fazer previsões para 2023 usando os dados de 2022
                predictions_rf = model_rf.predict(features_2022)

                # Criar uma tabela com as previsões do Random Forest
                tabela_previsoes_rf = pd.DataFrame({
                    'NOME': df_tratado['NOME'],
                    'INDE_2022': df_tratado['INDE_2022'],
                    'Previsão_INDE_2023_RF': predictions_rf
                })

                st.write(tabela_previsoes_rf)
                
                # Dados dos resultados
                models = ['Regressão Linear', 'Random Forest']
                mse_means = [np.mean(mse_scores_lr), np.mean(mse_scores_rf)]
                mse_stds = [np.std(mse_scores_lr), np.std(mse_scores_rf)]
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

                
                st.subheader("Comparação do Desempenho dos Modelos")
                
                st.markdown('')

                # Criar gráfico de barras para comparação de MSE médio
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(models, mse_means, yerr=mse_stds, capsize=10, color=['blue', 'green'])
                ax.set_title('Comparação do Erro Quadrático Médio (MSE) entre Modelos')
                ax.set_ylabel('Média do Erro Quadrático Médio (MSE)')
                ax.set_xlabel('Modelos')
                ax.set_ylim(0, max(mse_means) + 0.5)
                for i in range(len(models)):
                    ax.text(i, mse_means[i] + 0.02, f'{mse_means[i]:.4f}', ha='center', fontsize=12)
                ax.grid(True)
                st.pyplot(fig)
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                
                st.subheader('Interpretação dos Resultados:')
                
                st.markdown(' #### Regressão Linear:')
                
                st.markdown(' * Média do Erro Quadrático Médio (MSE): 0.6546')
                
                st.markdown('''O MSE médio da Regressão Linear é 0.6546. O MSE mede a média das diferenças ao quadrado entre os valores previstos e os valores reais. 
                            Quanto menor o MSE, melhor o modelo está ajustado aos dados. Nesse caso, o modelo de Regressão Linear tem um desempenho 
                            relativamente bom com um MSE médio abaixo de 1.''')
              
                st.markdown(' * Desvio Padrão do MSE: 0.2008')
                
                st.markdown('''O desvio padrão do MSE é 0.2008, indicando a variabilidade dos erros nas diferentes divisões dos dados (folds). 
                            Um desvio padrão mais baixo sugere que o modelo é mais consistente em diferentes partes dos dados.
                            ''')
                
                                
                st.markdown(' #### Random Forest:')
                
                st.markdown(' * Média do Erro Quadrático Médio (MSE): 0.6757')
                
                st.markdown('''O MSE médio do Random Forest é 0.6757, um pouco maior do que o MSE da Regressão Linear. Isso indica que, em termos de erro médio, 
                            o Random Forest não performou tão bem quanto a Regressão Linear para este conjunto de dados. 
                            No entanto, a diferença é pequena, então o desempenho dos dois modelos é comparável.
                            ''')
              
                st.markdown(' * Desvio Padrão do MSE: 0.1357')
                
                st.markdown('''O desvio padrão do MSE para o Random Forest é 0.1357, que é menor que o desvio padrão da Regressão Linear. 
                            Isso sugere que o Random Forest é mais consistente do que a Regressão Linear em termos de previsibilidade em diferentes partes dos dados.
                            ''')
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                st.markdown(' #### Comparação Geral:')
                
                st.markdown(''' * Precisão do Modelo: A Regressão Linear apresentou um MSE médio ligeiramente menor do que o Random Forest, sugerindo que, em média, 
                            a Regressão Linear produziu previsões mais próximas dos valores reais.                            
                            ''')
                
                st.markdown(''' * Consistência do Modelo: O Random Forest, por outro lado, apresentou um desvio padrão menor do MSE, o que indica que ele tende a ser mais 
                            consistente em diferentes divisões dos dados.
                            ''')
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                st.markdown(' #### Conclusão:')
                
                st.markdown(''' * Escolha do Modelo: Se o objetivo principal for a precisão, a Regressão Linear pode ser preferível devido ao seu MSE médio ligeiramente menor. 
                            No entanto, se a consistência for mais importante (ou seja, obter resultados mais uniformes em diferentes amostras dos dados), 
                            o Random Forest seria uma escolha melhor.                            
                            ''')
                
                st.markdown(''' * Diferenças Pequenas: As diferenças nos desempenhos dos dois modelos são relativamente pequenas, sugerindo que ambos os modelos são 
                            razoavelmente eficazes para este problema específico.                            
                            ''')
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                
                st.subheader(' Previsão com Regressão Linear')
                
                st.markdown(' * Comparativo INDE 2022 com previsão para 2023')
                
                
                 # 1. Box Plot Comparativo
                st.markdown("Box Plot")
                st.markdown('')
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=[tabela_previsoes_lr['INDE_2022'], tabela_previsoes_lr['Previsão_INDE_2023']], ax=ax)
                ax.set_xticklabels(['INDE 2022', 'Previsão INDE 2023'])
                ax.set_title('Box Plot do INDE 2022 e Previsão INDE 2023')
                st.pyplot(fig)
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)

                # 2. Histogramas Comparativos com Transparência
                st.markdown("Histogramas com Transparência")
                st.markdown('')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Ajustar transparência para evitar mistura de cores
                sns.histplot(tabela_previsoes_lr['INDE_2022'], color='blue', label='INDE 2022', kde=True, ax=ax, alpha=0.5)
                sns.histplot(tabela_previsoes_lr['Previsão_INDE_2023'], color='orange', label='Previsão INDE 2023', kde=True, ax=ax, alpha=0.5)

                ax.set_title('Distribuição do INDE 2022 e Previsão INDE 2023 com Transparência')
                ax.set_xlabel('INDE')
                ax.legend()
                st.pyplot(fig)
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
                st.subheader('Análise da Previsão')
                
                st.markdown(' #### Box Plot Comparativo')
                
                st.markdown('* INDE 2022 vs. Previsão INDE 2023:')   
                
                st.markdown('''Distribuição: O Box Plot mostra que a mediana do INDE 2022 é ligeiramente mais alta do que a mediana da Previsão do INDE 2023. 
                            Isso sugere que o modelo de Regressão Linear prevê uma leve diminuição nos valores do INDE para 2023 em comparação com 2022.
                            ''')    
                
                st.markdown('''Dispersão: A amplitude interquartil (a altura da caixa) do INDE 2022 é maior do que a da Previsão INDE 2023, 
                            indicando que os dados reais de 2022 foram mais dispersos em comparação com as previsões de 2023. 
                            Isso pode indicar que o modelo de regressão suavizou as previsões, reduzindo a variabilidade.
                            ''')       
                
                st.markdown('''Outliers: Ambos os box plots mostram a presença de outliers, mas o INDE 2022 parece ter mais outliers em valores baixos. 
                            Isso também pode indicar que os valores de 2022 tinham mais variação em seus extremos do que o previsto para 2023.''')
                
                st.markdown(' #### Histograma com Transparência')
                
                st.markdown('* Sobreposição de Distribuições:')
                
                st.markdown('''Semelhanças: O histograma mostra que as distribuições do INDE 2022 e da Previsão INDE 2023 têm uma forma semelhante, 
                            especialmente nas faixas centrais (entre 6 e 8), o que sugere que o modelo de Regressão Linear conseguiu capturar a estrutura básica dos dados de 2022.
                            ''')
                
                st.markdown('''Diferenças: No entanto, há uma leve diferença nas extremidades da distribuição. O INDE 2022 tem uma cauda à direita (valores mais altos) 
                            que parece estar menos presente na Previsão INDE 2023. Além disso, a densidade de previsões na faixa de 5 a 6 é maior do que no INDE 2022, 
                            o que sugere que o modelo está prevendo uma concentração maior de valores na faixa intermediária, com menos dispersão nas extremidades.''')
                
                
                st.markdown(' #### Conclusão')
                
                st.markdown('''O modelo de Regressão Linear parece estar suavizando as previsões para 2023 e parece capturar bem a distribuição central dos dados, 
                            no entanto isso resultou em uma menor variabilidade e menos extremos em comparação com os valores observados em 2022.''')
                
                st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
                
     
                
                
               


                
