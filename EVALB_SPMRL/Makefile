TESTFILE=dev.Arabic.gold.ptb.-feat
JAVA=java


all: evalb


clean:
	rm -f evalb_spmrl
	
install: evalb
	cp evalb_spmrl /usr/local/bin



evalb: evalb.c
	gcc -Wall -O3 -g -o evalb_spmrl evalb.c
	
evalb_linux: evalb.c
	gcc -Wall -fPIC -O3 -g -o evalb_spmrl evalb.c
#to compile on linux

          

# note: on the original makefile, the funsigned-char option was applied

home: install_home

install_home: all
	cp evalb_spmrl ${PREFIX}/bin
	
up:
	tar zcvf ../evalb_spmrl2013.tar.gz ../evalb_spmrl2013/
	putW ../evalb_spmrl2013.tar.gz



#################################
# stuff to debug some treebanks #
#################################	
test_full: all
	./evalb dev.Arabic.gold.ptb dev.Arabic.gold.ptb


test: all
	./evalb -p ./new.prm ${TESTFILE} ${TESTFILE}

debug: all
	./evalb -D ${TESTFILE} ${TESTFILE}
	echo "./evalb -D ${TESTFILE} ${TESTFILE}"

debug_one: all
	lines 616 < ${TESTFILE} > ${TESTFILE}.616
	./evalb -D ${TESTFILE}.616 ${TESTFILE}.616
	echo "./evalb -D ${TESTFILE}.616 ${TESTFILE}.616"	

releaf:
	./evalb -D dev.Arabic.gold.ptb.-feat.616.bug dev.Arabic.gold.ptb.-feat.616.bug
	echo "./evalb -D dev.Arabic.gold.ptb.-feat.616.bug dev.Arabic.gold.ptb.-feat.616.bug" > /dev/stderr

java: 
	${JAVA} -jar ./evalC/evalC.jar ${TESTFILE} ${TESTFILE} /dev/stdout
		
	
		