import requests
import zipfile
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def download_single_file(i, start, end, output_dir, base_url, print_lock, stats):
    """
    Télécharge un seul fichier TWIC
    
    Args:
        i: Numéro TWIC
        start, end: Plage pour affichage
        output_dir: Dossier de destination
        base_url: URL de base
        print_lock: Lock pour l'affichage thread-safe
        stats: Dictionnaire partagé pour les statistiques
    """
    filename = f"twic{i}g.zip"
    filepath = os.path.join(output_dir, filename)
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(filepath):
        with print_lock:
            print(f"[{i-start+1}/{end-start+1}] {filename} - Déjà téléchargé (ignoré)")
            stats['skipped'] += 1
        return
    
    url = f"{base_url}{i}g.zip"
    
    # En-têtes pour simuler un navigateur
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            file_size = len(response.content) / 1024  # Taille en KB
            with print_lock:
                print(f"[{i-start+1}/{end-start+1}] {filename} - Téléchargé ({file_size:.1f} KB)")
                stats['successful'] += 1
        else:
            with print_lock:
                print(f"[{i-start+1}/{end-start+1}] {filename} - Erreur HTTP {response.status_code}")
                stats['failed'] += 1
            
    except requests.exceptions.RequestException as e:
        with print_lock:
            print(f"[{i-start+1}/{end-start+1}] {filename} - Erreur: {str(e)}")
            stats['failed'] += 1


def download_twic_files(start=920, end=1617, output_dir="twic_files", max_workers=6):
    """
    Télécharge les fichiers PGN du site The Week In Chess en parallèle
    
    Args:
        start: Numéro de début (défaut: 920)
        end: Numéro de fin (défaut: 1617)
        output_dir: Dossier de destination (défaut: "twic_files")
        max_workers: Nombre de threads parallèles (défaut: 6)
    
    Returns:
        tuple: (nombre de succès, nombre d'échecs, nombre ignorés)
    """
    # Créer le dossier de destination s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://theweekinchess.com/zips/twic"
    
    # Statistiques partagées entre threads
    stats = {'successful': 0, 'failed': 0, 'skipped': 0}
    print_lock = Lock()
    
    print("="*60)
    print(f"ÉTAPE 1/2 : TÉLÉCHARGEMENT DES FICHIERS")
    print("="*60)
    print(f"Téléchargement de {end - start + 1} fichiers TWIC ({start} à {end})...")
    print(f"Dossier de destination: {output_dir}")
    print(f"Téléchargement parallèle sur {max_workers} threads\n")
    
    # Téléchargement parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(start, end + 1):
            future = executor.submit(
                download_single_file, 
                i, start, end, output_dir, base_url, print_lock, stats
            )
            futures.append(future)
        
        # Attendre la fin de tous les téléchargements
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"Erreur inattendue: {str(e)}")
    
    print("\n" + "-"*60)
    print(f"Téléchargement terminé!")
    print(f"Succès: {stats['successful']} | Échecs: {stats['failed']} | Ignorés: {stats['skipped']}")
    print("-"*60 + "\n")
    
    return stats['successful'], stats['failed'], stats['skipped']


def unzip_and_merge_pgn(zip_dir="twic_files", output_file="archive.pgn", keep_extracted=True, pgn_dir="pgn_files"):
    """
    Dézippe tous les fichiers ZIP TWIC et fusionne les PGN en un seul fichier
    
    Args:
        zip_dir: Dossier contenant les fichiers ZIP (défaut: "twic_files")
        output_file: Nom du fichier PGN final (défaut: "archive.pgn")
        keep_extracted: Garder les fichiers PGN extraits (défaut: True)
        pgn_dir: Dossier pour sauvegarder les PGN individuels (défaut: "pgn_files")
    
    Returns:
        tuple: (nombre traités, nombre échoués, total parties)
    """
    
    print("="*60)
    print(f"ÉTAPE 2/2 : DÉZIPAGE ET FUSION DES FICHIERS PGN")
    print("="*60)
    
    if not os.path.exists(zip_dir):
        print(f"Erreur: Le dossier '{zip_dir}' n'existe pas!")
        return 0, 0, 0
    
    # Créer le dossier pour les PGN individuels si on veut les garder
    if keep_extracted:
        Path(pgn_dir).mkdir(parents=True, exist_ok=True)
    
    # Créer un dossier temporaire pour l'extraction
    extract_dir = os.path.join(zip_dir, "temp_extracted")
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    
    # Récupérer tous les fichiers ZIP et les trier
    zip_files = sorted([f for f in os.listdir(zip_dir) if f.endswith('.zip')])
    
    if not zip_files:
        print(f"Aucun fichier ZIP trouvé dans '{zip_dir}'")
        return 0, 0, 0
    
    print(f"Traitement de {len(zip_files)} fichiers ZIP...\n")
    
    processed = 0
    failed = 0
    total_games = 0
    
    # Ouvrir le fichier de sortie en mode écriture
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as output:
        
        for idx, zip_file in enumerate(zip_files, 1):
            zip_path = os.path.join(zip_dir, zip_file)
            
            try:
                # Extraire le fichier ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Trouver le fichier PGN extrait
                pgn_files = [f for f in os.listdir(extract_dir) if f.endswith('.pgn')]
                
                if not pgn_files:
                    print(f"[{idx}/{len(zip_files)}] {zip_file} - Aucun fichier PGN trouvé")
                    failed += 1
                    continue
                
                # Lire et fusionner chaque fichier PGN
                games_in_file = 0
                for pgn_file in pgn_files:
                    pgn_path = os.path.join(extract_dir, pgn_file)
                    
                    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn:
                        content = pgn.read()
                        
                        # Compter approximativement le nombre de parties
                        games = content.count('[Event ')
                        games_in_file += games
                        total_games += games
                        
                        # Écrire dans le fichier de sortie
                        output.write(content)
                        
                        # Ajouter une ligne vide entre les fichiers
                        if not content.endswith('\n\n'):
                            output.write('\n\n')
                    
                    # Copier le fichier PGN dans le dossier des PGN individuels si demandé
                    if keep_extracted:
                        dest_path = os.path.join(pgn_dir, pgn_file)
                        os.rename(pgn_path, dest_path)
                    else:
                        # Sinon supprimer le fichier temporaire
                        os.remove(pgn_path)
                
                print(f"[{idx}/{len(zip_files)}] {zip_file} - Fusionné ({games_in_file} parties)")
                processed += 1
                
            except zipfile.BadZipFile:
                print(f"[{idx}/{len(zip_files)}] {zip_file} - Fichier ZIP corrompu")
                failed += 1
            except Exception as e:
                print(f"[{idx}/{len(zip_files)}] {zip_file} - Erreur: {str(e)}")
                failed += 1
    
    # Nettoyer le dossier d'extraction temporaire
    try:
        os.rmdir(extract_dir)
    except:
        pass
    
    print("\n" + "-"*60)
    print(f"Fusion terminée!")
    print(f"Fichiers traités: {processed} | Fichiers échoués: {failed}")
    print(f"Total de parties: ~{total_games}")
    if keep_extracted:
        print(f"PGN individuels sauvegardés dans: {pgn_dir}")
    print("-"*60 + "\n")
    
    return processed, failed, total_games


def main(start=920, end=1617, output_dir="twic_files", output_file="archive.pgn", keep_extracted=True, pgn_dir="pgn_files", max_workers=6):
    """
    Fonction principale: télécharge, dézippe et fusionne tous les fichiers TWIC
    
    Args:
        start: Numéro de début TWIC
        end: Numéro de fin TWIC
        output_dir: Dossier pour les fichiers ZIP
        output_file: Nom du fichier PGN final
        keep_extracted: Garder les fichiers PGN extraits individuellement
        pgn_dir: Dossier pour sauvegarder les PGN individuels
        max_workers: Nombre de threads pour le téléchargement parallèle
    """
    print("\n" + "="*60)
    print("TÉLÉCHARGEMENT ET FUSION DES ARCHIVES TWIC")
    print("="*60 + "\n")
    
    # Étape 1: Téléchargement parallèle
    dl_success, dl_failed, dl_skipped = download_twic_files(start, end, output_dir, max_workers)
    
    # Étape 2: Dézipage et fusion
    merge_processed, merge_failed, total_games = unzip_and_merge_pgn(
        zip_dir=output_dir,
        output_file=output_file,
        keep_extracted=keep_extracted,
        pgn_dir=pgn_dir
    )
    
    # Résumé final
    print("="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"Fichiers téléchargés: {dl_success}")
    print(f"Fichiers déjà présents: {dl_skipped}")
    print(f"Échecs téléchargement: {dl_failed}")
    print(f"Fichiers fusionnés: {merge_processed}")
    print(f"Échecs fusion: {merge_failed}")
    print(f"Total de parties d'échecs: ~{total_games}")
    print(f"Fichier fusionné: {output_file}")
    if keep_extracted:
        print(f"PGN individuels: {pgn_dir}/")
    
    # Taille du fichier final
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"Taille du fichier fusionné: {file_size:.2f} MB")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Lancer le processus complet avec 6 threads et sauvegarde des PGN individuels
    main(start=920, end=1617, max_workers=6, keep_extracted=True)
    
    # Exemples de personnalisation:
    # main(start=1000, end=1100, max_workers=6)  # Plage personnalisée
    # main(output_file="mes_parties.pgn", max_workers=8)  # 8 threads
    # main(keep_extracted=False, max_workers=6)  # Ne pas garder les PGN individuels
    # main(pgn_dir="mes_pgn", max_workers=6)  # Dossier personnalisé pour les PGN