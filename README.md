# CamShift OCL #

Implementacja algorytmu CamShift z wykorzystaniem OpenCL (liczenie histogramu, obliczanie momentów obrazu).

-----
### Wymagania ###

* Karta graficzna zgodna z OpenCL 1.1 (lub 1.2) i wgrane odpowiednie dla urządzenia sterowniki (program był testowany na AMD Radeon HD 7770).
* OpenCV (testowane na wersji 2.4.9).
* Podłączona kamerka (testowane na rozdzielczości 640x480).

-----
### Budowanie ###

**Windows:**

* Utworzyć projekt z Visual Studio 12 (lub nowszym).
* Dodać ścieżki do nagłówków OpenCL (cl.hpp wymagane) i nagłówków OpenCV do projektu (C/C++/General/Additional Include Directories).
* Dodać bibliotekę OpenCL.lib do projektu (Linker/Input/Additional Dependencies).
* Dodać ścieżkę do bibliotek OpenCV (Linker/General/Additional Library Directories).

**Linux:**

* TBD

**! Uwaga dla użytkowników NVIDIA:**

* Zapewne będzie brakowało pliku [cl.hpp wersja 1.2](http://www.khronos.org/registry/cl/api/1.2/cl.hpp) i umieścić tam, gdzie reszta nagłówków OpenCL.
* Jednak nie testowałem tego, więc w razie problemów użyć [cl.hpp w wersji 1.1](http://www.khronos.org/registry/cl/api/1.1/cl.hpp) i zamienić "cl::Local" na "cl::__local" w CamShift.cpp (ta opcja była testowana).

-----
### Używanie ###

* Naciśnięcie spacji powoduje rozpoczęcie śledzenia obiektu widocznego w prostokącie.
* Ponowne naciśnięcie spacji zatrzymuje śledzenie.
* Naciskając ESC wychodzimy z programu.

-----
### GL & HF ###

Pierwszy projekt z OpenCL, więc uwagi mile widziane.