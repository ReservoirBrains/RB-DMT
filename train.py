import argparse
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import math
from tqdm.auto import tqdm

# Assure-toi que l'import correspond à l'arborescence de ton projet
from DT.DynamicalTransformer import DynamicalTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search DynamicalTransformer sur données HCP")
    
    # --- Matériel & Reproductibilité ---
    parser.add_argument('--device', type=str, default='cuda:0', help='Device PyTorch (ex: cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=0, help='Seed aléatoire pour la reproductibilité')
    
    # --- Données & Chemins ---
    parser.add_argument('--data_dir', type=str, default='data/schaefer200', help='Dossier contenant les CSV schaefer200')
    parser.add_argument('--targets_file', type=str, default='data/targets.csv', help='Fichier CSV des cibles')
    parser.add_argument('--out_temp', type=str, default='results_grid_search_temp.csv', help='Nom du fichier de sauvegarde temporaire')
    parser.add_argument('--out_final', type=str, default='results_grid_search_final.csv', help='Nom du fichier de sauvegarde final')
    parser.add_argument('--nb_train', type=int, default=800, help='Nombre de sujets pour le Train')
    parser.add_argument('--nb_test', type=int, default=100, help='Nombre de sujets pour le Test')
    
    # --- Grille de recherche ---
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 3e-4, 1e-4], help='Liste des Learning Rates à tester')
    parser.add_argument('--targets', type=str, nargs='+', default=None, help='Liste spécifique de cibles (par défaut: liste complète du script)')
    
    # --- Hyperparamètres d'entraînement ---
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques maximum')
    parser.add_argument('--batch_size', type=int, default=4, help='Taille de batch VRAM (batch_size physique)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Étapes d\'accumulation (Batch effectif = batch_size * grad_accum)')
    parser.add_argument('--patience', type=int, default=5, help='Patience pour le Early Stopping')
    parser.add_argument('--chunk_size', type=int, default=256, help='Chunk size pour le DynamicalTransformer')
    
    # --- Hyperparamètres du Modèle ---
    parser.add_argument('--layers', type=int, default=4, help='Nombre de couches')
    parser.add_argument('--m_units', type=int, default=32, help='Memory units')
    parser.add_argument('--m_dim', type=int, default=512, help='Memory dimension')
    parser.add_argument('--a_dim', type=int, default=128, help='Attention dimension')
    parser.add_argument('--a_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Taux de dropout')
    parser.add_argument('--d_conv', type=int, default=1, help='Dimension de convolution')

    return parser.parse_args()


def load_data(args):
    # Définition du nom du fichier de sauvegarde en fonction de la seed et des tailles
    cache_filename = f"data_prep_{args.seed}_{args.nb_train}_{args.nb_test}.npz"
    
    # ==========================================
    # 1. TENTATIVE DE CHARGEMENT DEPUIS LE CACHE
    # ==========================================
    if os.path.exists(cache_filename):
        print(f"\n📂 Fichier de cache trouvé : {cache_filename}")
        print("⏳ Chargement des données en mémoire...")
        
        # On charge le fichier compressé
        cached_data = np.load(cache_filename)
        
        X_TRAIN_BASE = cached_data['X_TRAIN_BASE']
        X_TEST_BASE = cached_data['X_TEST_BASE']
        # On reconvertit les tableaux numpy en listes Python pour les IDs
        train_subject_ids = cached_data['train_subject_ids'].tolist()
        test_subject_ids = cached_data['test_subject_ids'].tolist()
        
        print(f"✅ Données récupérées depuis le cache ! Train: {X_TRAIN_BASE.shape}, Test: {X_TEST_BASE.shape}")
        return X_TRAIN_BASE, X_TEST_BASE, train_subject_ids, test_subject_ids

    # ==========================================
    # 2. AUCUN CACHE -> LECTURE DES CSV
    # ==========================================
    print(f"\n🔍 Aucun cache trouvé ({cache_filename}). Scan du dossier {args.data_dir} et filtrage des IDs...")
    filenames = os.listdir(args.data_dir)
    all_ids = [x.split("_")[0] for x in filenames]
    valid_ids = [i for i in np.unique(all_ids) if all_ids.count(i) == 2]

    np.random.seed(args.seed)
    np.random.shuffle(valid_ids)

    if args.nb_test + args.nb_train > len(valid_ids):
        raise Exception(f"Pas assez de sujets. Requis : {args.nb_test + args.nb_train}, Dispo : {len(valid_ids)}")

    X_TRAIN_RAW, train_subject_ids = [], []
    X_TEST_RAW, test_subject_ids = [], []

    print("Chargement des séries temporelles (Train)...")
    for _id in tqdm(valid_ids[:args.nb_train], desc="Extraction X_TRAIN"):
        try:
            name1 = f"{_id}_REST1_parcellated_timeseries_schaefer200.csv"
            name2 = f"{_id}_REST2_parcellated_timeseries_schaefer200.csv"
            d1 = pd.read_csv(f'{args.data_dir}/{name1}', header=None).to_numpy()
            d2 = pd.read_csv(f'{args.data_dir}/{name2}', header=None).to_numpy()
        except:
            continue
        
        if d1.shape == (200, 2400):
            X_TRAIN_RAW.append(d1)
            train_subject_ids.append(int(_id))
        if d2.shape == (200, 2400):
            X_TRAIN_RAW.append(d2)
            train_subject_ids.append(int(_id))

    print("Chargement des séries temporelles (Test)...")
    for _id in tqdm(valid_ids[args.nb_train: args.nb_train + args.nb_test], desc="Extraction X_TEST"):
        try:
            name1 = f"{_id}_REST1_parcellated_timeseries_schaefer200.csv"
            name2 = f"{_id}_REST2_parcellated_timeseries_schaefer200.csv"
            d1 = pd.read_csv(f'{args.data_dir}/{name1}', header=None).to_numpy()
            d2 = pd.read_csv(f'{args.data_dir}/{name2}', header=None).to_numpy()
        except:
            continue
            
        if d1.shape == (200, 2400):
            X_TEST_RAW.append(d1)
            test_subject_ids.append(int(_id))
        if d2.shape == (200, 2400):
            X_TEST_RAW.append(d2)
            test_subject_ids.append(int(_id))

    X_TRAIN_BASE = np.array(X_TRAIN_RAW).transpose(0, 2, 1)
    X_TEST_BASE = np.array(X_TEST_RAW).transpose(0, 2, 1)

    print(f"✅ X chargés depuis les CSV ! Train: {X_TRAIN_BASE.shape}, Test: {X_TEST_BASE.shape}")

    # ==========================================
    # 3. SAUVEGARDE DU NOUVEAU CACHE
    # ==========================================
    # print(f"💾 Sauvegarde des données compilées dans '{cache_filename}' (Compression en cours)...")
    # np.savez_compressed(
    #     cache_filename,
    #     X_TRAIN_BASE=X_TRAIN_BASE,
    #     X_TEST_BASE=X_TEST_BASE,
    #     train_subject_ids=np.array(train_subject_ids),
    #     test_subject_ids=np.array(test_subject_ids)
    # )
    # print("✅ Cache sauvegardé avec succès !")

    return X_TRAIN_BASE, X_TEST_BASE, train_subject_ids, test_subject_ids


def run_experiment(target_var, lr, args, all_targets, X_TRAIN_BASE, X_TEST_BASE, train_subject_ids, test_subject_ids):
    print(f"\n{'='*60}")
    print(f"🚀 LANCEMENT : Target = {target_var} | LR = {lr}")
    print(f"{'='*60}")

    # 1. Alignement des Y
    print("  -> Extraction et alignement des variables cibles...")
    if target_var not in all_targets.columns:
        print(f"  ⚠️ La cible {target_var} n'existe pas dans le fichier CSV. Expérience ignorée.")
        return None

    target_data = all_targets[target_var]
    Y_TRAIN, Y_TEST = [], []
    
    valid_train_indices = []
    for i, _id in enumerate(train_subject_ids):
        y_val = target_data[all_targets["Subject"] == _id].to_numpy()
        if len(y_val) > 0 and not np.isnan(y_val[0]):
            Y_TRAIN.append(y_val[0])
            valid_train_indices.append(i)
            
    valid_test_indices = []
    for i, _id in enumerate(test_subject_ids):
        y_val = target_data[all_targets["Subject"] == _id].to_numpy()
        if len(y_val) > 0 and not np.isnan(y_val[0]):
            Y_TEST.append(y_val[0])
            valid_test_indices.append(i)

    if len(Y_TRAIN) == 0 or len(Y_TEST) == 0:
        print(f"  ⚠️ Pas assez de données valides pour {target_var}. Expérience ignorée.")
        return None

    print(f"  -> Sujets conservés (sans NaN) : {len(valid_train_indices)} en Train | {len(valid_test_indices)} en Test")

    # Filtrage des X correspondants
    X_TRAIN_LOCAL = X_TRAIN_BASE[valid_train_indices]
    X_TEST_LOCAL = X_TEST_BASE[valid_test_indices]
    Y_TRAIN = np.array(Y_TRAIN)
    Y_TEST = np.array(Y_TEST)

    # 2. Normalisation Z-Score
    print("  -> Normalisation Z-score des données en cours...")
    X_mean = X_TRAIN_LOCAL.mean(axis=(0, 1), keepdims=True)
    X_std = X_TRAIN_LOCAL.std(axis=(0, 1), keepdims=True) + 1e-8
    X_TRAIN_LOCAL = (X_TRAIN_LOCAL - X_mean) / X_std
    X_TEST_LOCAL = (X_TEST_LOCAL - X_mean) / X_std

    Y_mean = Y_TRAIN.mean()
    Y_std = Y_TRAIN.std() + 1e-8
    Y_TRAIN = (Y_TRAIN - Y_mean) / Y_std
    Y_TEST = (Y_TEST - Y_mean) / Y_std
    y_std_val = float(Y_std)

    # 3. Préparation PyTorch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.tensor(X_TRAIN_LOCAL, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_TRAIN, dtype=torch.float32)
    X_test_t = torch.tensor(X_TEST_LOCAL, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_TEST, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=args.batch_size, shuffle=False)

    num_features = X_train_t.shape[2] 
    output_dim = 1

    # 4. Initialisation du Modèle
    params = {
        'num_layers': args.layers,
        'memory_units': args.m_units,
        'memory_dim': args.m_dim,
        'attention_dim': args.a_dim,
        'attention_heads': args.a_heads,
        'dropout': args.dropout,
        'd_conv': args.d_conv,
        'input_dim': num_features,  
        'output_dim': output_dim,   
        'seq_len': X_TRAIN_LOCAL.shape[1],
        'device': device,
    }

    model = DynamicalTransformer(**params)
    model.input_projection = nn.Linear(num_features, params['attention_dim']).to(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    # 5. Boucle d'entraînement
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    print(f"  -> Début de l'entraînement ({args.epochs} époques max, Patience={args.patience})...")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        
        # On s'assure que les gradients sont à zéro au début de l'époque
        optimizer.zero_grad()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            out = model(x_batch, chunk_size=args.chunk_size) 
            logits = out.mean(dim=1) 
            
            logits = logits.view(-1)
            y_batch = y_batch.view(-1)
            
            # Calcul de la loss
            loss = criterion(logits, y_batch)
            
            # Division de la loss par le nombre d'accumulations pour moyenner correctement
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            
            # On accumule la "vraie" loss (non divisée) pour les logs
            total_train_loss += loss.item()

            # On fait un pas d'optimisation uniquement tous les `grad_accum` batchs, ou à la fin du dataloader
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        
        if np.isnan(avg_train_loss):
            print(f"    [Epoch {epoch+1:02d}/{args.epochs}] ⚠️ Arrêt forcé : La loss d'entraînement est NaN.")
            break

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val, chunk_size=args.chunk_size)
                logits_val = out_val.mean(dim=1).view(-1)
                y_val = y_val.view(-1)
                val_loss = criterion(logits_val, y_val)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)

        print(f"    [Epoch {epoch+1:02d}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ", end="")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print("--> 🌟 Nouveau meilleur modèle !")
        else:
            patience_counter += 1
            print(f"--> Patience : {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"  🛑 Early stopping déclenché à l'époque {epoch+1}.")
            break

    # 6. Évaluation finale
    print("  -> Évaluation finale du meilleur modèle sur le Test Set...")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    model.eval()
    test_sq_error, test_abs_error = 0.0, 0.0
    test_sum_y, test_sum_sq_y = 0.0, 0.0  
    total_test = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch, chunk_size=args.chunk_size)
            logits = out.mean(dim=1).view(-1)
            y_batch = y_batch.view(-1)
            
            test_abs_error += torch.abs(logits - y_batch).sum().item()
            test_sq_error += ((logits - y_batch)**2).sum().item() 
            test_sum_y += y_batch.sum().item()
            test_sum_sq_y += (y_batch**2).sum().item()
            total_test += y_batch.size(0)

    test_rmse_norm = math.sqrt(test_sq_error / total_test) 
    test_mae_norm = test_abs_error / total_test
    
    test_rmse_reel = test_rmse_norm * y_std_val
    test_mae_reel = test_mae_norm * y_std_val
    
    mean_y = test_sum_y / total_test
    var_y = (test_sum_sq_y / total_test) - (mean_y**2)
    std_y = math.sqrt(max(0, var_y))
    test_srmse = test_rmse_norm / std_y if std_y > 0 else float('inf')

    print(f"  ✅ RÉSULTAT FINAL | RMSE: {test_rmse_reel:.4f} | SRMSE: {test_srmse:.4f} | MAE: {test_mae_reel:.4f}")

    return {
        'Target': target_var,
        'LR': lr,
        'SRMSE': test_srmse,
        'RMSE': test_rmse_reel,
        'MAE': test_mae_reel
    }


def main():
    args = parse_args()
    
    print(f"--- Configuration Globale ---")
    print(f"Device: {args.device}")
    print(f"Batch Effectif: {args.batch_size * args.grad_accum} (Batch Physique: {args.batch_size} x Accumulation: {args.grad_accum})")
    print(f"Patience: {args.patience} | Epochs: {args.epochs}")
    print(f"Fichiers Out: {args.out_temp} / {args.out_final}")

    # Définition des cibles
    default_targets = [
        # "PicVocab_Unadj", "AngAggr_Unadj", "ReadEng_Unadj", "PosAffect_Unadj",
        # "WM_Task_Acc", "SCPT_SPEC", "NEOFAC_O", "PercStress_Unadj",
        # "DDisc_AUC_40K", "PercReject_Unadj", "Relational_Task_Acc", "Social_Task_Perc_TOM",
        # "PMAT24_A_CR", "AngHostil_Unadj", "VSPLOT_TC", "ER40ANG",
        # "Endurance_Unadj", "EmotSupp_Unadj", "NEOFAC_E", "PSQI_Score",
        # "CardSort_Unadj", "SelfEff_Unadj", "PicSeq_Unadj", "NEOFAC_A",
        # "Language_Task_Story_Avg_Difficulty_Level", "PainInterf_Tscore",
        "MeanPurp_Unadj", "FearAffect_Unadj", "ProcSpeed_Unadj", "ER40_CR",
        "LifeSatisf_Unadj", "Friendship_Unadj", "Sadness_Unadj", "ER40NOE",
        "Flanker_Unadj", "NEOFAC_N", "ListSort_Unadj", "Taste_Unadj",
        "InstruSupp_Unadj", "FearSomat_Unadj", "Language_Task_Math_Avg_Difficulty_Level",
        "ER40HAP", "ER40SAD", "ER40FEAR", "Strength_Unadj", "AngAffect_Unadj",
        "IWRD_TOT", "Social_Task_Perc_Random", "Loneliness_Unadj", "MMSE_Score",
        "PercHostil_Unadj", "SCPT_SEN", "Dexterity_Unadj", "Odor_Unadj",
        "NEOFAC_C", "Emotion_Task_Face_Acc", "GaitSpeed_Comp", "Mars_Final"
    ]
    
    # On utilise la liste custom si fournie via CLI, sinon la liste par défaut
    targets_to_run = args.targets if args.targets is not None else default_targets

    # 1. Chargement des données cibles
    all_targets = pd.read_csv(args.targets_file)

    # 2. Chargement lourd des séries temporelles
    X_TRAIN_BASE, X_TEST_BASE, train_subject_ids, test_subject_ids = load_data(args)

    # 3. Lancement du Grid Search
    results = []
    for target in tqdm(targets_to_run, desc="Progression globale (Cibles)"):
        for lr in args.lrs:
            res = run_experiment(
                target_var=target, 
                lr=lr, 
                args=args,
                all_targets=all_targets,
                X_TRAIN_BASE=X_TRAIN_BASE, 
                X_TEST_BASE=X_TEST_BASE, 
                train_subject_ids=train_subject_ids, 
                test_subject_ids=test_subject_ids
            )
            
            if res is not None:
                results.append(res)
                # Sauvegarde intermédiaire
                pd.DataFrame(results).to_csv(args.out_temp, index=False)

    # 4. Synthèse finale
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.out_final, index=False)

    print(f"\n🎉 Expériences terminées ! Sauvegardé dans '{args.out_final}'")
    
    # Affichage du meilleur tableau récap
    if not df_results.empty:
        best_results = df_results.loc[df_results.groupby('Target')['SRMSE'].idxmin()]
        print("\nMeilleurs résultats par cible :")
        print(best_results[['Target', 'LR', 'SRMSE', 'RMSE']])


if __name__ == "__main__":
    main()