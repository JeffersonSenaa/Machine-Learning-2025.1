import sqlite3
import pandas as pd
import os
from pathlib import Path

def main():
    # Verificar se o arquivo CSV existe
    csv_path = 'campeonato-brasileiro-full.csv'
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado!")
        return

    try:
        # Create connection
        conn = sqlite3.connect('Camp.db')
        cursor = conn.cursor()
        print("Conexão com banco de dados estabelecida com sucesso!")

        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS brasileirao (
                ID INTEGER,
                rodata INTEGER,
                data TEXT,
                hora TEXT,
                mandante TEXT,
                visitante TEXT,
                formacao_mandante TEXT,
                formacao_visitante TEXT,
                tecnico_mandante TEXT,
                tecnico_visitante TEXT,
                vencedor TEXT,
                arena TEXT,
                mandante_Placar INTEGER,
                visitante_Placar INTEGER,
                mandante_Estado TEXT,
                visitante_Estado TEXT
            )
        ''')
        print("Tabela criada/verificada com sucesso!")

        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"CSV carregado com sucesso! Total de registros: {len(df)}")

        # Insert data into table
        df.to_sql('brasileirao', conn, if_exists='replace', index=False)
        print("Dados inseridos com sucesso!")

        # Verificar se os dados foram inseridos corretamente
        cursor.execute("SELECT COUNT(*) FROM brasileirao")
        count = cursor.fetchone()[0]
        print(f"Total de registros na tabela: {count}")

        # Exemplo de consulta para verificar os dados
        cursor.execute("SELECT * FROM brasileirao LIMIT 5")
        print("\nPrimeiros 5 registros importados:")
        for row in cursor.fetchall():
            print(row)

    except sqlite3.Error as e:
        print(f"Erro no SQLite: {e}")
    except pd.errors.EmptyDataError:
        print("Erro: O arquivo CSV está vazio!")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    finally:
        if 'conn' in locals():
            conn.commit()
            conn.close()
            print("\nConexão fechada.")

if __name__ == "__main__":
    main()


