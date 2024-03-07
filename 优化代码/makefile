CC=/usr/sw/swgcc/swgcc710-tools-SEA-1374/usr/bin/sw9gcc
CXX=/usr/sw/swgcc/swgcc710-tools-SEA-1374/usr/bin/sw9g++

CFLAGS= -mslave -msimd -mieee -mfma -funroll-all-loops -Ofast
CXXFLAGS= -mhost -mieee -mftz -fpermissive -msimd -funroll-all-loops -Ofast
INCLUDE=-I.

EXE=pcg_solve

all: $(EXE)

$(EXE): main.o pcg.o slave.o
	$(CXX) -mhybrid -o$(EXE) $^ -L. -lpcg_solve

main.o:	main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ 

pcg.o: pcg.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@	

slave.o: slave.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(EXE) *.o;
