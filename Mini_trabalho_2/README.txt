Projeto: Previsão de Resultados do Brasileirão com Machine Learning  
Mini Trabalho 2 – Aquisição de Dados  

1. Fontes dos dados:  
- Base 1: Brazilian Soccer Database (Kaggle)  
  Link: https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro 
  Licença: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)  

- Base 2: Campeonato Brasileiro de futebol (GitHub - adaoduque/Brasileirao_Dataset)  
  Link: https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol  
  Licença: Repositório público com código aberto para fins educacionais e analíticos.  

- Base 3: Brazilian Soccer - Brasileirão - Brasileirao 
  Link: https://www.kaggle.com/datasets/gustavomartino/brasileirao?resource=download
  Licença: Licença aberta para uso não comercial e educacional, disponibilizada no repositório GitHub.

2. Justificativa da escolha:  
As bases foram escolhidas por conterem dados históricos completos de partidas do Campeonato Brasileiro, incluindo informações como placares, mandante/visitante, estatísticas dos times, datas e rodadas. Essas informações são altamente relevantes para a construção de modelos preditivos no contexto do nosso projeto de Aprendizado de Máquina, cujo objetivo é prever os resultados das partidas do Brasileirão.

3. Descrição do conteúdo:  
As bases incluem colunas como data da partida, times mandante e visitante, número de gols, estatísticas de desempenho e posição na tabela, entre outras variáveis que podem ser úteis para modelagem preditiva.

As três bases apresentam dados sobre o Campeonato Brasileiro, com diferentes formatos e níveis de detalhe:

- A Base 1 contém colunas como datetime, home_team, away_team, home_goal, away_goal, season, round, entre outras.

- A Base 2 detalha informações como: rodata, data, hora, mandante, visitante, formação do mandante, formação do visitante, técnico do mandante, técnico do visitante, vencedor, arena, placar do mandante, placar do visitante, estado do mandante e estado do visitante.

- A Base 3 apresenta dados similares às demais, com foco em partidas do Brasileirão, incluindo colunas como: round, day, month, year, season, home team, goals HT, goals VT e visiting team.

Essas informações são fundamentais para a construção de variáveis independentes e dependentes, permitindo o desenvolvimento de modelos de aprendizado supervisionado.

4. Participantes:
- Harryson Campos Martins — Matrícula: 211039466  
- Pedro Henrique Muniz de Oliveira — Matrícula: 200059947  
- Flávio Gustavo Araújo de Melo — Matrícula: 211030602  
- Leandro de Almeida Oliveira — Matrícula: 211030827  
- Jefferson Sena Oliveira — Matrícula: 200020323  
- José André Rabelo Rocha — Matrícula: 211062016  

5. Ferramentas e recursos utilizados:
A coleta foi feita diretamente através do site Kaggle e GitHub. O processo de escolha das bases, análise preliminar e observações relevantes estão documentados em um arquivo PDF que acompanha esta entrega. Esse documento fornece detalhes adicionais sobre como os dados foram baixados, o conteúdo analisado e considerações iniciais sobre possíveis pré-processamentos.

6. Questões éticas e legais:  
As bases utilizadas estão publicamente disponíveis no Kaggle e possuem licenças compatíveis com uso acadêmico. Não há dados pessoais ou informações sensíveis nas bases. O uso se dá exclusivamente para fins educacionais, respeitando os princípios éticos e legais da coleta de dados.

7. Estrutura dos arquivos entregues:
- README.txt → este arquivo.  
- MiniTrabalho2.pdf → Documento com análise das bases, justificativas e detalhes do processo de coleta.  
- Nome do dataset ->  Dataset e ferramenta ou método utilizado para extrair e armazenar os dados de forma segura. 
