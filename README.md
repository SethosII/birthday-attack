# birthday-attack
Birthday attack on a hash function using CUDA for the module "Kryptologie" at the HTWK Leipzig.

## Task-Description

> Aufgabe 10: Geburtstagsangriff
Der Geburtstagsangriff auf Streufunktionen soll implementiert werden. Eine
Streufunktion mit 32 Bit Streuwertlänge soll gebildet werden durch bitweises
EXOR der acht 32-Bit-Blöcke des Ergebnisses der SHA-256 Streufunktion.
Diese Streufunktion soll mit einem Geburtstagsangriff folgendermaßen ge-
knackt werden. Schreiben Sie zwei gegensätzliche Rezensionen zu einen frei
gewählten künstlerischen Ereignis. Beide Rezensionen sollen 16 Schablonen
enthalten, die mit je 2 möglichen Worten belegt werden können. Bestimmen
Sie zu jeder Belegung die Streuwerte und testen Sie auf Kollisionen.
Mögliche Variante: Geburtstagsangriff mit CUDA. 64 Bit Streuwertlänge.
Implementierung auf einem Graphikprozessor (CUDA-Framework).