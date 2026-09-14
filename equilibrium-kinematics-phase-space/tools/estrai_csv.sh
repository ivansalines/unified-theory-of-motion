#!/bin/bash

# Intestazione del file CSV
echo "Numberline_Q,Type,Step,E,Q,dt,||F||" > risultati.csv

# Ciclo sui file ordinati numericamente
for file in $(printf "%s\n" *_Q*.log | sort -V); do
    # Estrae l'ID numerico dal nome del file (es. 749 da esecuzione_749.log)
    # id=$(echo $file | grep -oE '[0-9]+')
    id=$(echo "$file" | grep -oP '(?<=Q)[0-9]+')    
    line=""
    type=""

    # Controllo Condizione 2: Converged
    if grep -q "Converged" "$file"; then
        line=$(grep "Converged" "$file")
        type="Converged"
    # Controllo Condizione Stiffness (cerca la riga precedente a quella con 'stiffness')
    elif grep -iq "stiffness" "$file"; then
        line=$(grep -B 1 -i "stiffness" "$file" | head -n 1)
        type="Stiffness"
    fi

    # Se abbiamo trovato una riga valida, estraiamo i dati
    if [ -n "$line" ]; then
        # La regex [[:space:]]* gestisce l'eventuale spazio dopo l'uguale
        step=$(echo "$line" | sed -nE 's/.*step=[[:space:]]*([0-9]+).*/\1/p')
        E=$(echo "$line" | sed -nE 's/.*E=[[:space:]]*([e0-9.+\-]*).*/\1/p' | cut -d',' -f1)
        Q=$(echo "$line" | sed -nE 's/.*Q=[[:space:]]*([e0-9.+\-]*).*/\1/p' | cut -d',' -f1)
        dt=$(echo "$line" | sed -nE 's/.*dt=[[:space:]]*([e0-9.+\-]*).*/\1/p' | cut -d',' -f1)
        F=$(echo "$line" | sed -nE 's/.*\|\|F\|\|=[[:space:]]*([e0-9.+\-]*).*/\1/p')

        # Scrittura nel file CSV (rimuovendo eventuali spazi residui con tr)
        echo "$id,$type,$(echo $step | tr -d ' '),$(echo $E | tr -d ' '),$(echo $Q | tr -d ' '),$(echo $dt | tr -d ' '),$(echo $F | tr -d ' ')" >> risultati.csv
    fi
done

echo "Processo completato. I dati sono in 'risultati.csv'."
