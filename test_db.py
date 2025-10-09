import sqlite3
import random
from datetime import datetime, timedelta


def crea_database():
    # Connessione al database (lo crea se non esiste)
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()

    # Elimina le tabelle se esistono già (per ricrearle da capo)
    cursor.execute('DROP TABLE IF EXISTS dettagli_ordine')
    cursor.execute('DROP TABLE IF EXISTS ordini')
    cursor.execute('DROP TABLE IF EXISTS inventario')
    cursor.execute('DROP TABLE IF EXISTS prodotti')
    cursor.execute('DROP TABLE IF EXISTS fornitori')
    cursor.execute('DROP TABLE IF EXISTS categorie')

    # 1. Tabella CATEGORIE
    cursor.execute('''
    CREATE TABLE categorie (
        id_categoria INTEGER PRIMARY KEY AUTOINCREMENT,
        nome_categoria TEXT NOT NULL,
        descrizione TEXT
    )
    ''')

    # 2. Tabella FORNITORI
    cursor.execute('''
    CREATE TABLE fornitori (
        id_fornitore INTEGER PRIMARY KEY AUTOINCREMENT,
        nome_fornitore TEXT NOT NULL,
        contatto TEXT,
        telefono TEXT,
        email TEXT,
        indirizzo TEXT,
        citta TEXT,
        paese TEXT
    )
    ''')

    # 3. Tabella PRODOTTI
    cursor.execute('''
    CREATE TABLE prodotti (
        id_prodotto INTEGER PRIMARY KEY AUTOINCREMENT,
        nome_prodotto TEXT NOT NULL,
        descrizione TEXT,
        prezzo_unitario REAL NOT NULL,
        id_categoria INTEGER,
        id_fornitore INTEGER,
        codice_prodotto TEXT UNIQUE,
        FOREIGN KEY (id_categoria) REFERENCES categorie(id_categoria),
        FOREIGN KEY (id_fornitore) REFERENCES fornitori(id_fornitore)
    )
    ''')

    # 4. Tabella INVENTARIO
    cursor.execute('''
    CREATE TABLE inventario (
        id_inventario INTEGER PRIMARY KEY AUTOINCREMENT,
        id_prodotto INTEGER,
        quantita_disponibile INTEGER NOT NULL,
        quantita_minima INTEGER NOT NULL,
        data_ultimo_aggiornamento DATE,
        FOREIGN KEY (id_prodotto) REFERENCES prodotti(id_prodotto)
    )
    ''')

    # 5. Tabella ORDINI
    cursor.execute('''
    CREATE TABLE ordini (
        id_ordine INTEGER PRIMARY KEY AUTOINCREMENT,
        data_ordine DATE NOT NULL,
        id_fornitore INTEGER,
        stato_ordine TEXT CHECK(stato_ordine IN ('Pendente', 'Confermato', 'Spedito', 'Ricevuto', 'Annullato')),
        quantita_totale INTEGER NOT NULL,
        totale_ordine REAL,
        note TEXT,
        FOREIGN KEY (id_fornitore) REFERENCES fornitori(id_fornitore)
    )
    ''')

    # 6. Tabella DETTAGLI_ORDINE
    cursor.execute('''
    CREATE TABLE dettagli_ordine (
        id_dettaglio INTEGER PRIMARY KEY AUTOINCREMENT,
        id_ordine INTEGER,
        id_prodotto INTEGER,
        quantita INTEGER NOT NULL,
        prezzo_unitario REAL NOT NULL,
        subtotale REAL,
        FOREIGN KEY (id_ordine) REFERENCES ordini(id_ordine),
        FOREIGN KEY (id_prodotto) REFERENCES prodotti(id_prodotto)
    )
    ''')

    print("Tabelle create con successo!")
    return conn, cursor


def inserisci_dati(cursor):
    # Inserimento CATEGORIE
    categorie_data = [
        ('Elettronica', 'Dispositivi elettronici e accessori'),
        ('Abbigliamento', 'Vestiti e accessori moda'),
        ('Casa e Giardino', 'Articoli per la casa e il giardinaggio'),
        ('Sport e Tempo Libero', 'Attrezzature sportive e ricreative'),
        ('Libri e Media', 'Libri, DVD, CD e prodotti multimediali'),
        ('Alimentari', 'Prodotti alimentari e bevande'),
        ('Salute e Bellezza', 'Prodotti per la cura personale'),
        ('Automotive', 'Ricambi e accessori per auto'),
        ('Giocattoli', 'Giochi e giocattoli per bambini'),
        ('Ufficio', 'Forniture per ufficio e cancelleria'),
        ('Elettrodomestici', 'Grandi e piccoli elettrodomestici'),
        ('Musica', 'Strumenti musicali e accessori'),
        ('Fotografia', 'Fotocamere e accessori fotografici'),
        ('Informatica', 'Computer e componenti informatici'),
        ('Arredamento', 'Mobili e complementi d\'arredo'),
        ('Giardinaggio', 'Attrezzi e piante per giardino'),
        ('Ferramenta', 'Attrezzi e materiale per bricolage'),
        ('Cucina', 'Utensili e accessori da cucina'),
        ('Bagno', 'Accessori e prodotti per il bagno'),
        ('Illuminazione', 'Lampade e sistemi di illuminazione')
    ]

    cursor.executemany('INSERT INTO categorie (nome_categoria, descrizione) VALUES (?, ?)', categorie_data)

    # Inserimento FORNITORI
    fornitori_data = [
        ('TechnoItalia SRL', 'Marco Rossi', '02-12345678', 'info@technoitalia.it', 'Via Roma 15', 'Milano', 'Italia'),
        ('ModaStyle SpA', 'Laura Bianchi', '06-87654321', 'vendite@modastyle.it', 'Via del Corso 45', 'Roma', 'Italia'),
        ('CasaBella Import', 'Giuseppe Verde', '051-555444', 'ordini@casabella.it', 'Via Indipendenza 78', 'Bologna',
         'Italia'),
        ('SportMax Distribution', 'Andrea Neri', '011-333222', 'sport@sportmax.it', 'Corso Francia 120', 'Torino',
         'Italia'),
        ('MediaWorld Supply', 'Francesca Gialli', '041-777888', 'media@mediaworld.it', 'Fondamenta Nuove 34', 'Venezia',
         'Italia'),
        ('GustoItalia Food', 'Roberto Blu', '055-444555', 'food@gustoitalia.it', 'Via dei Calzaiuoli 67', 'Firenze',
         'Italia'),
        ('BellezzaPura Cosmetics', 'Silvia Rosa', '081-666777', 'beauty@bellezzapura.it', 'Via Toledo 89', 'Napoli',
         'Italia'),
        (
        'AutoParts Center', 'Davide Grigio', '010-999888', 'auto@autoparts.it', 'Via del Porto 23', 'Genova', 'Italia'),
        ('ToysLand Import', 'Elena Viola', '0721-111222', 'toys@toysland.it', 'Viale della Repubblica 56', 'Pesaro',
         'Italia'),
        ('OfficeSupply Pro', 'Matteo Arancio', '0532-333444', 'office@officesupply.it', 'Corso Giovecca 12', 'Ferrara',
         'Italia'),
        (
        'ElettroMax SpA', 'Carla Verde', '049-555666', 'elettro@elettromax.it', 'Via del Santo 78', 'Padova', 'Italia'),
        ('MusicPlanet Store', 'Luca Oro', '0542-777999', 'music@musicplanet.it', 'Via Emilia 145', 'Imola', 'Italia'),
        ('PhotoPro Equipment', 'Sara Argento', '0731-222333', 'photo@photopro.it', 'Corso Mazzini 67', 'Senigallia',
         'Italia'),
        ('ComputerLand Tech', 'Paolo Rame', '0571-444666', 'computer@computerland.it', 'Via San Lorenzo 34', 'Empoli',
         'Italia'),
        ('ArredoCasa Design', 'Anna Bronzo', '0565-888999', 'arredo@arredocasa.it', 'Viale Italia 90', 'Piombino',
         'Italia'),
        ('GreenGarden Supply', 'Marco Ferro', '0575-111333', 'garden@greengarden.it', 'Via Guido Monaco 23', 'Arezzo',
         'Italia'),
        ('Ferramenta Moderna', 'Lisa Acciaio', '0583-555777', 'ferramenta@moderna.it', 'Via Fillungo 56', 'Lucca',
         'Italia'),
        ('CucinaStyle Import', 'Diego Platino', '0586-222444', 'cucina@cucinastyle.it', 'Via Grande 78', 'Livorno',
         'Italia'),
        (
        'BagnoLux Design', 'Marta Titanio', '050-666888', 'bagno@bagnolux.it', 'Lungarno Mediceo 45', 'Pisa', 'Italia'),
        ('LightDesign Pro', 'Fabio Zinco', '0577-333555', 'light@lightdesign.it', 'Piazza del Campo 12', 'Siena',
         'Italia')
    ]

    cursor.executemany(
        'INSERT INTO fornitori (nome_fornitore, contatto, telefono, email, indirizzo, citta, paese) VALUES (?, ?, ?, ?, ?, ?, ?)',
        fornitori_data)

    # Inserimento PRODOTTI (almeno 20)
    prodotti_data = [
        ('Smartphone Samsung Galaxy', 'Ultimo modello Android', 599.99, 1, 1, 'SMSG001'),
        ('Laptop Dell Inspiron', 'Notebook 15.6 pollici', 899.99, 14, 14, 'LTDL002'),
        ('T-shirt cotone biologico', 'Maglietta unisex eco-friendly', 25.50, 2, 2, 'TSHC003'),
        ('Jeans slim fit', 'Denim stretch blu scuro', 89.90, 2, 2, 'JNSF004'),
        ('Aspirapolvere senza fili', 'Potenza 150W, batteria litio', 199.99, 11, 11, 'ASPV005'),
        ('Macchina caffè espresso', 'Automatica con macinacaffè', 449.99, 18, 18, 'MCAF006'),
        ('Pallone da calcio FIFA', 'Ufficiale per competizioni', 35.00, 4, 4, 'PLFC007'),
        ('Racchetta da tennis', 'Professionale fibra carbonio', 159.99, 4, 4, 'RCTN008'),
        ('Romanzo bestseller', 'Ultimo thriller internazionale', 18.50, 5, 5, 'RMBR009'),
        ('Olio extravergine oliva', 'Bottiglia 750ml DOP', 12.90, 6, 6, 'OLEV010'),
        ('Pasta integrale biologica', 'Confezione 500g grano duro', 3.50, 6, 6, 'PSIN011'),
        ('Crema viso antietà', 'Formula con acido ialuronico', 45.00, 7, 7, 'CRVS012'),
        ('Shampoo capelli grassi', 'Flacone 250ml purificante', 8.90, 7, 7, 'SHCG013'),
        ('Filtro aria auto', 'Compatibile motori benzina', 28.50, 8, 8, 'FLAR014'),
        ('Bambola interattiva', 'Con suoni e movimenti', 67.99, 9, 9, 'BMBI015'),
        ('Puzzle 1000 pezzi', 'Paesaggio montano', 15.90, 9, 9, 'PZ1K016'),
        ('Stampante laser', 'Monocromatica A4 veloce', 179.99, 10, 10, 'STPL017'),
        ('Risme carta A4', 'Confezione 5 risme 80g/mq', 22.50, 10, 10, 'RSCA018'),
        ('Frigorifero combinato', 'Classe A++ 300 litri', 599.99, 11, 11, 'FRGC019'),
        ('Microonde digitale', 'Potenza 900W con grill', 129.99, 11, 11, 'MCRD020'),
        ('Chitarra acustica', 'Legno massello con custodia', 289.99, 12, 12, 'CHAC021'),
        ('Fotocamera reflex', 'Digitale 24MP con obiettivo', 899.99, 13, 13, 'FTRF022'),
        ('Mouse gaming RGB', 'Ottico 3200 DPI programmabile', 49.99, 14, 14, 'MSGR023'),
        ('Divano 3 posti', 'Tessuto grigio sfoderabile', 699.99, 15, 15, 'DVN3024'),
        ('Tavolo da pranzo', 'Legno massello 6 posti', 459.99, 15, 15, 'TVPR025')
    ]

    cursor.executemany(
        'INSERT INTO prodotti (nome_prodotto, descrizione, prezzo_unitario, id_categoria, id_fornitore, codice_prodotto) VALUES (?, ?, ?, ?, ?, ?)',
        prodotti_data)

    # Inserimento INVENTARIO
    inventario_data = []
    for i in range(1, 26):  # Per ogni prodotto
        quantita_disp = random.randint(5, 200)
        quantita_min = random.randint(5, 20)
        data_agg = datetime.now() - timedelta(days=random.randint(1, 30))
        inventario_data.append((i, quantita_disp, quantita_min, data_agg.strftime('%Y-%m-%d')))

    cursor.executemany(
        'INSERT INTO inventario (id_prodotto, quantita_disponibile, quantita_minima, data_ultimo_aggiornamento) VALUES (?, ?, ?, ?)',
        inventario_data)

    # Inserimento ORDINI
    stati_ordine = ['Pendente', 'Confermato', 'Spedito', 'Ricevuto', 'Annullato']
    ordini_data = []
    for i in range(25):  # 25 ordini
        data_ordine = datetime.now() - timedelta(days=random.randint(1, 90))
        id_fornitore = random.randint(1, 20)
        stato = random.choice(stati_ordine)
        quantita_tot = random.randint(5, 100)  # Quantità totale dell'ordine
        totale = round(random.uniform(50.0, 2000.0), 2)
        note = f"Ordine numero {i + 1} - Priorità normale"
        ordini_data.append((data_ordine.strftime('%Y-%m-%d'), id_fornitore, stato, quantita_tot, totale, note))

    cursor.executemany(
        'INSERT INTO ordini (data_ordine, id_fornitore, stato_ordine, quantita_totale, totale_ordine, note) VALUES (?, ?, ?, ?, ?, ?)',
        ordini_data)

    # Inserimento DETTAGLI_ORDINE
    dettagli_data = []
    for ordine_id in range(1, 26):  # Per ogni ordine
        num_prodotti = random.randint(1, 5)  # Ogni ordine ha da 1 a 5 prodotti diversi
        prodotti_selezionati = random.sample(range(1, 26), num_prodotti)

        for prodotto_id in prodotti_selezionati:
            quantita = random.randint(1, 10)
            prezzo_unit = round(random.uniform(10.0, 500.0), 2)
            subtotale = round(quantita * prezzo_unit, 2)
            dettagli_data.append((ordine_id, prodotto_id, quantita, prezzo_unit, subtotale))

    cursor.executemany(
        'INSERT INTO dettagli_ordine (id_ordine, id_prodotto, quantita, prezzo_unitario, subtotale) VALUES (?, ?, ?, ?, ?)',
        dettagli_data)

    print("Tutti i dati sono stati inseriti con successo!")


def mostra_statistiche(cursor):
    print("\n" + "=" * 50)
    print("STATISTICHE DATABASE MAGAZZINO")
    print("=" * 50)

    # Conta record per tabella
    tabelle = ['categorie', 'fornitori', 'prodotti', 'inventario', 'ordini', 'dettagli_ordine']
    for tabella in tabelle:
        cursor.execute(f'SELECT COUNT(*) FROM {tabella}')
        count = cursor.fetchone()[0]
        print(f"{tabella.capitalize()}: {count} record")

    print("\n" + "-" * 30)
    print("ESEMPI DI QUERY")
    print("-" * 30)

    # Query di esempio 1: Prodotti con scorte basse
    print("\n1. Prodotti con scorte sotto la soglia minima:")
    cursor.execute('''
    SELECT p.nome_prodotto, i.quantita_disponibile, i.quantita_minima
    FROM prodotti p
    JOIN inventario i ON p.id_prodotto = i.id_prodotto
    WHERE i.quantita_disponibile < i.quantita_minima
    LIMIT 5
    ''')

    risultati = cursor.fetchall()
    for row in risultati:
        print(f"  - {row[0]}: {row[1]} disponibili (min: {row[2]})")

    # Query di esempio 2: Ordini per fornitore
    print("\n2. Totale ordini per fornitore (top 5):")
    cursor.execute('''
    SELECT f.nome_fornitore, COUNT(o.id_ordine) as num_ordini, 
           SUM(o.quantita_totale) as quantita_ordinata,
           ROUND(SUM(o.totale_ordine), 2) as totale_speso
    FROM fornitori f
    JOIN ordini o ON f.id_fornitore = o.id_fornitore
    GROUP BY f.id_fornitore, f.nome_fornitore
    ORDER BY totale_speso DESC
    LIMIT 5
    ''')

    risultati = cursor.fetchall()
    for row in risultati:
        print(f"  - {row[0]}: {row[1]} ordini, {row[2]} pezzi, €{row[3]}")


def main():
    print("Creazione database magazzino...")
    conn, cursor = crea_database()

    print("Inserimento dati...")
    inserisci_dati(cursor)

    # Salva le modifiche
    conn.commit()

    # Mostra statistiche
    mostra_statistiche(cursor)

    # Chiude la connessione
    conn.close()
    print(f"\nDatabase 'test.db' creato con successo!")
    print("Puoi ora utilizzarlo con qualsiasi strumento SQLite.")


if __name__ == "__main__":
    main()