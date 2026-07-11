# KestrelNet: Route-Aware Freight Scheduling via Curriculum Reinforcement Learning

**Meridian Institute Technical Report MI-TR-2031-07 — published 18 June 2031**

## Abstract

KestrelNet is a route-aware freight scheduling system developed at the
Meridian Institute. It learns dispatch policies for mixed road-rail cargo
networks using curriculum reinforcement learning: agents are first trained
on single-corridor toy networks and graduate to the full Meridian
metropolitan graph only after meeting punctuality gates. On the Harborline
benchmark, KestrelNet reduced average cargo dwell time by 31 percent
against the tabu-search baseline while holding on-time delivery above
96 percent. The system's distinguishing contribution is the punctuality
curriculum itself: scheduling quality emerged from staged task difficulty
rather than from reward shaping, and the same curriculum transferred to
the smaller Solstice depot network without retraining.

## Contents

1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1.1 Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Summary of Results . . . . . . . . . . . . . . . . . . . . . . . . . 4
2 Approach . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.1 The Punctuality Curriculum . . . . . . . . . . . . . . . . . . . . . 5
2.2 Reward Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2.3 KestrelNet-Zero: Learning Without Demonstrations . . . . . . . . . . 8
3 Evaluation on the Harborline Benchmark . . . . . . . . . . . . . . . . 10
3.1 Dwell Time and Punctuality . . . . . . . . . . . . . . . . . . . . . 10
3.2 Transfer to the Solstice Depot Network . . . . . . . . . . . . . . . 12
4 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.1 Failure Cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
4.2 Unsuccessful Attempts . . . . . . . . . . . . . . . . . . . . . . . . 15
5 Conclusion, Limitations, and Future Work . . . . . . . . . . . . . . . 16
A Contributions and Acknowledgments . . . . . . . . . . . . . . . . . . 18
B References . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

## 1. Introduction

Freight dispatch in dense metropolitan networks remains dominated by
hand-tuned heuristics. The Meridian Institute's scheduling group set out
to test whether a learned policy could beat the institute's production
tabu-search dispatcher on its own network without inheriting its rule
base. KestrelNet is the result: a policy network trained end to end with
curriculum reinforcement learning, named after the kestrel's ability to
hold position over a moving landscape.

The project produced two systems. KestrelNet proper is trained with a
staged curriculum and a small set of dispatcher demonstrations for cold
start. KestrelNet-Zero omits the demonstrations entirely and learns from
network simulation alone; it matches KestrelNet's final dwell-time numbers
but takes 3.4 times as many simulation steps and produces schedules that
human dispatchers described as harder to read.

### 1.1 Contributions

First, we show that a punctuality-gated curriculum — agents may not
progress to a harder network until on-time delivery exceeds 95 percent on
the current one — removes the need for elaborate reward shaping. Second,
we document KestrelNet-Zero, demonstrating that demonstration-free
training is feasible at higher compute cost. Third, we release the
Harborline benchmark: 90 days of anonymized cargo movements over a
419-node road-rail graph.

## 2. Approach

### 2.1 The Punctuality Curriculum

Training proceeds over five network stages: single corridor, twin
corridor, ring, district, and the full metropolitan graph. Each stage
must be passed at 95 percent on-time delivery before the next unlocks.
Early in training the agent exploits a loophole — holding low-priority
cargo indefinitely to keep expresses punctual — which disappears at the
district stage where dwell-time penalties activate. We observed a
characteristic transition the team called the patience shift: at the ring
stage, agents spontaneously begin reserving platform slack ahead of
predicted congestion rather than reacting to it.

### 2.2 Reward Design

The reward is deliberately sparse: a terminal punctuality score, a dwell
penalty, and nothing else. No shaping terms for routing choices, platform
assignment, or coupling order. Ablations that added intermediate shaping
rewards consistently produced worse transfer to the Solstice network.

### 2.3 KestrelNet-Zero

KestrelNet-Zero begins from a randomly initialized policy with no
demonstrations. For the first two stages its schedules are chaotic;
punctuality stays below 40 percent for 11 million steps. The breakthrough
arrives abruptly at the twin-corridor stage, after which progress mirrors
the demonstration-seeded run. The final policy is equal on dwell time,
2.1 points worse on schedule readability as scored by a dispatcher panel.

## 3. Evaluation on the Harborline Benchmark

### 3.1 Dwell Time and Punctuality

On Harborline's held-out 30-day slice, KestrelNet reduces mean cargo
dwell time from 6.8 hours (tabu-search baseline) to 4.7 hours, a 31
percent reduction, with on-time delivery at 96.4 percent versus the
baseline's 94.1. Peak-hour improvements are larger: dwell falls 38
percent in the 07:00–09:00 window.

### 3.2 Transfer to the Solstice Depot Network

Applied unchanged to the Solstice depot network — one ninth the size,
different rolling stock — KestrelNet holds a 24 percent dwell-time
advantage. The curriculum, not the network weights, appears to carry the
transferable structure: retraining from scratch on Solstice alone reaches
the same numbers only when the curriculum is preserved.

## 4. Discussion

### 4.1 Failure Cases

KestrelNet mishandles simultaneous multi-terminal disruptions: when three
or more terminals close at once, the policy oscillates between reroutes
for up to twenty simulated minutes. The production tabu dispatcher, for
all its rigidity, degrades more gracefully under compound disruption.

### 4.2 Unsuccessful Attempts

Process-reward models over intermediate schedule states, Monte-Carlo
tree search over dispatch actions, and dense per-decision rewards were
all tried and abandoned; each either overfit the metropolitan graph or
collapsed the policy into imitating the tabu baseline.

## 5. Conclusion, Limitations, and Future Work

A punctuality-gated curriculum is sufficient to train a freight dispatch
policy that beats a production heuristic dispatcher on its home network.
The main limitations are compound-disruption behavior and the simulation
fidelity gap: Harborline models mechanical failures as terminal closures,
which is coarser than reality. Future work targets disruption-robust
curricula and joint passenger-freight scheduling with the Meridian
transit authority.

## Appendix A. Contributions and Acknowledgments

### Core Contributors

Astrid Vennemo
Bao-Long Trinh
Cecilie Andrada
Darius Okonjo
Eleni Vasquez-Moreno
Fenna Lindqvist
Gustav Reinholt
Hana Okabe
Imre Csordas
Jelena Marovic
Kofi Adjei-Mensah
Liv Sondergaard
Matteo Chiellini
Noor al-Rashidi
Oskar Wennerstrom
Priya Raghunathan
Quentin Delacroix
Rosa Beltran-Ibarra
Sander Vollan
Tuva Eikeland

### Contributors

Adaeze Nwosu
Aksel Brattbakk
Alva Sjogren
Amara Diallo
Anders Kvistad
Aniela Wozniak
Arne Solheim
Asta Norregaard
Beatrix Hollander
Bendik Aarseth
Birgitta Lundgren
Bjorn Tessand
Camille Roussel
Carsten Wiig
Chidi Okafor
Clara Vestergaard
Dagny Hoyland
Damla Yilmazer
Dilnoza Karimova
Edvard Bruland
Eirik Sandmo
Elzbieta Gorska
Emeka Obiora
Enzo Marchetti
Erlend Vikanes
Esperanza Gutierrez
Fabiola Mendes
Farrukh Toshev
Filippa Ohlsson
Franciszek Zielinski
Frida Ramstad
Gaute Lillevik
Giulia Ferrante
Greta Nyström
Gunnar Thorsen
Halvor Espedal
Hedda Brekkan
Heikki Rantanen
Henrik Dalgaard
Ibrahim Sowe
Ida Kleppestad
Ilaria Bonetti
Ingvild Rosnes
Irena Havlickova
Isabella Cortez-Ruiz
Ivar Lokken
Jakub Novotny
Jasmina Begic
Joakim Sundelin
Johanna Vikström
Jonas Eltoft
Josefina Aguirre
Julius Baumgartner
Kaja Vinterbo
Kalle Jokinen
Karim Bouzidi
Katarzyna Lis
Kirsten Aalberg
Klara Jensdottir
Knut Havardsen
Kristoffer Melgaard
Lars-Erik Bjornstad
Laura Kaczmarek
Leif Gundersen
Lena Holmqvist
Leon Kaufmann
Lidia Ferreira
Linnea Bergfalk
Lorenzo Vitale
Lucas Meier
Lukas Prochazka
Magda Szymanska
Magnus Rydberg
Maja Lindholm
Malin Osterberg
Marek Dvorak
Margrete Skarsvag
Mariana Pacheco
Marius Grindvoll
Marta Kowalczyk
Mathilde Brenna
Mats Ekelund
Michal Cerny
Mikkel Andreassen
Milena Petronijevic
Mirek Blazek
Mona Fagerland
Nadia Berrada
Natalia Sokolowska
Nikolaj Friis
Nils Hagström
Nina Ravndal
Olav Systad
Oliwia Mazur
Omar Benjelloun
Otso Makela
Paulina Wisniewska
Pavel Horak
Pernille Vangsnes
Petter Ohman
Rafal Kubiak
Ragnhild Osteras
Rasmus Wickman
Renata Vargova
Rikke Thomassen
Roberta Colombo
Ruben Fjellheim
Sanna Koskinen
Sara Lindegren
Sebastian Falk
Sigrid Morkved
Silje Haaland
Simona Dvorakova
Sofia Marinescu
Solveig Bratten
Stefan Kowalski
Stig Rasmussen
Sunniva Loken
Svein Odegaard
Tarjei Vollmo
Teodora Ilic
Terje Skogland
Thea Gjersvik
Tomasz Lewandowski
Tor-Erik Sandvik
Tove Ellingsen
Ulrik Norheim
Vanja Kristoffersen
Veronika Slama
Viktor Lindeberg
Vilde Aasheim
Wanda Jablonska
Wojciech Kaminski
Yasmina El-Fassi
Yngve Torvund
Zofia Adamczyk
Zuzana Bartosova

### Acknowledgments

The Harborline benchmark was assembled with the cooperation of the
Meridian transit authority's freight division. Compute was provided by
the institute's Skarn cluster.

## Appendix B. References

Andrada, C. and Vennemo, A. (2029). Tabu search at scale: a decade of
the Meridian dispatcher. Journal of Transit Operations, 44(2).

Brattbakk, A., Okabe, H., and Trinh, B.-L. (2030). Punctuality gates for
staged network curricula. Proceedings of the Conference on Learned
Logistics.

Csordas, I. (2028). Reward hacking in depot simulators: seven case
studies. Meridian Institute Technical Report MI-TR-2028-11.

Diallo, A. and Sondergaard, L. (2030). The Harborline data model.
Meridian Institute Technical Report MI-TR-2030-03.

Lindqvist, F., Reinholt, G., and Marovic, J. (2029). Schedule
readability as a first-class metric. Workshop on Human-Centered
Dispatch.

Okonjo, D. (2031). Curriculum transfer between freight networks of
unequal scale. Proceedings of the Conference on Learned Logistics.

Raghunathan, P. and Vasquez-Moreno, E. (2030). Demonstration-free
dispatch learning. Meridian Institute Technical Report MI-TR-2030-09.

Vollan, S., Eikeland, T., and Delacroix, Q. (2031). The patience shift:
emergent slack reservation in staged scheduling curricula. Journal of
Transit Operations, 46(1).
