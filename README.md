# birthday-attack

This project implements the [birthday attack](https://en.wikipedia.org/wiki/Birthday_attack) on a hash function using CUDA for the module "Kryptologie" at the HTWK Leipzig. The hash function is derived from SHA-256 by XOR of the eight 32-bit blocks of SHA-256. Here is an example:

```
input:
This is a great day.

SHA-256 (hex)
93784ef2 2ab7c997 d0026ee8 8c2b37d0 5cdfd410 d41b84fb 90653697 e471d6e6

SHA-256 (binary)
1001 0011 0111 1000 0100 1110 1111 0010
0010 1010 1011 0111 1100 1001 1001 0111
1101 0000 0000 0010 0110 1110 1110 1000
1000 1100 0010 1011 0011 0111 1101 0000
0101 1100 1101 1111 1101 0100 0001 0000
1101 0100 0001 1011 1000 0100 1111 1011
1001 0000 0110 0101 0011 0110 1001 0111
1110 0100 0111 0001 1101 0110 1110 0110

derived hash via XOR (binary):
0001 1001 0011 0110 0110 1110 1100 0111

derived hash via XOR (hex):
19366ec7
```

[![](http://imgs.xkcd.com/comics/protocol.png)](http://xkcd.com/1323/ "xkcd.com")

A few words about the background: Charlie has to send a report to Bob and Alice should express her conformance via a digital signature. She also gets a copy from Charlie and sends the hash value encrypted to Bob. Bob can calculate the hash of his copy and validate Alice signature. Charlie needs two different texts with the same hash value to trick them. Therefore he needs two texts with some stencils to vary them and eventually find a collision.

Here is the output off an example run on a Nvidia Tesla K20 with the code compiled via `nvcc -arch=sm_35 -rdc=true -maxrregcount=64 -O3 src/main.cu src/sha256.cu src/birthdayAttack.cu -o birthday-attack`:

```
DEBUG GPU:
DEBUG: testSha256LongInput passed
DEBUG: testReduceSha256 passed
This birthday attack took 165.9 ms.
Collisions found: 2


Collision with good text #17470 and bad text #21144

Good plaintext:
Linux is awesome! It is build on top of reliable Software like X11. Also there are several projects within the Linux community which have mostly the same objective like Wayland and Mir - and that's great. Why shouldn't you? Duplicate work is no problem. And with different approaches you will propably find better solutions. Thanks to the many projects you can choose what suits you best. The next point are the codenames. They rule. How awesome was Heisenbug? Also the whole development process is amazing. Some people argue that the development lacks focus. They have thousands of unpaid developers throwing code, part time, into a giant, internet-y vat of software. And it works. As an old saying tells us: "the more the merrier". Sometimes even Linux users themself rant about Linux - and they still love it because they can critically examine. That's why everybody should use Linux!

Bad plaintext:
Linux sucks! It is build on top of very old Software like X11 which makes them hard to maintain. Also there are several projects within the Linux community which have mostly identical aim like Wayland and Mir. Therefore there is duplicate work. This shows how divided the community is. The next point are the codenames. They suck. What should Trusty Thar stand for? Also the whole development process sucks. They have thousands of unpaid developers throwing code, part time, into a giant, internet-y tub of software. What could possibly go wrong? The only result can be a giant pile of crap. The development lacks focus. As an old saying goes: "too many cooks spoil the broth". So the freedom for the users consists of choices between lots of semi-finished projects. Even Linux users themself rant about it. That's why nobody should use Linux!

Hash value of both is: 640cb7c7


Collision with good text #51660 and bad text #26306

Good plaintext:
Linux is awesome! It is build on top of reliable Software like X11. Also there are several projects within the Linux community which have mostly identical aim like Wayland and Mir - and that's great. Why shouldn't you? Duplicate work is no problem. And with more approaches you will propably find better solutions. Because of the many projects you can choose what fits you best. The next point are the code names. They rule. How awesome was Beefy Miracle? Also the whole development process is amazing. Some people argue that there is no focus in the development. They have thousands of unpaid developers throwing code, part time, into a giant, internet-y tub of software. And it works. As an old saying tells us: "the more the merrier". Sometimes even Linux users themself rant about Linux - and they still love it because they can critically examine. Therefore everybody should use Linux!

Bad plaintext:
Linux sucks! It is build on top of very old Software like X11 which makes them difficult to maintain. Also there are several projects within the Linux community which have mostly identical aim like Wayland and Mir. Because of such things there is a duplication of effort. This shows how divided the community is. The next point are the code names. They suck. What should Trusty Thar stand for? Also the whole development process sucks. They have thousands of unpaid developers throwing code, part time, into a giant, internet-y tub of software. What could possibly go wrong? The only result can be a gigantic pile of crap. There is no focus in the development. As an old proverb goes: "too many cooks spoil the broth". So the freedom for the users consists of choices between lots of semi-finished projects. Even Linux users themself rant about it. That's why nobody should use Linux!

Hash value of both is: 91ada029
```

The plaintexts are inspired by [Bryan Lunduke](http://lunduke.com/)'s talk [Linux sucks](http://www.youtube.com/watch?v=5pOxlazS3zs).

## Task-Description

> Aufgabe 10: Geburtstagsangriff  
> Der Geburtstagsangriff auf Streufunktionen soll implementiert werden. Eine Streufunktion mit 32 Bit Streuwertlänge soll gebildet werden durch bitweises EXOR der acht 32-Bit-Blöcke des Ergebnisses der SHA-256 Streufunktion. Diese Streufunktion soll mit einem Geburtstagsangriff folgendermaßen geknackt werden. Schreiben Sie zwei gegensätzliche Rezensionen zu einen frei gewählten künstlerischen Ereignis. Beide Rezensionen sollen 16 Schablonen enthalten, die mit je 2 möglichen Worten belegt werden können. Bestimmen Sie zu jeder Belegung die Streuwerte und testen Sie auf Kollisionen. Mögliche Variante: Geburtstagsangriff mit CUDA. 64 Bit Streuwertlänge. Implementierung auf einem Graphikprozessor (CUDA-Framework).
