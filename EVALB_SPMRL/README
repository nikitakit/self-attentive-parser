
// DjamŽ: version record added for history's sake.
// note to future updater: please add your changelog below

(copied from http://nlp.cs.nyu.edu/evalb/ )
EVALB20080701.tgz (July 1, 2008 version) modified by Don Blaheta (Knox College)
EVALB20060307.tgz (March 3, 2006 version; debuged of Jan. 17, 2006 version) modified by David Ellis (Brown University)
EVALB20060117.tgz (Jan. 17, 2006 version) modified by David Ellis (Brown University)
EVALB20050908.tgz (Sept. 8, 2005 version) modified by David Brroks (Birmingham)
EVALB.tgz (original version).
Authors

Satoshi Sekine (New York University) :  e-mail: his last name (at) cs.nyu.edu
Michael John Collins (University of Pennsylvania)
Note: the authors are not responsible for the newer versions. We put these versions even without checking the program. Please be responsible for yourself.

*************************************************************************

Modification

David Brroks (Birmingham): fixed the code so that the program can be compiled by the latest gcc (September 2005). Helps are given by Peet Morris and Ramon Ziai through the Corpora Mailing list.
David Ellis (Brown University) : fixes a bug in which sentences were incorrectly categorized as "length mismatch" when the the parse output had certain mislabeled parts-of-speech.
Don Blaheta (KNOX) : fixes a bug on the output of last number of the total information was not TOTAL_crossing, but it was TOTAL_non_crossing.



April 2012
// Modified by Slav Petrov and Ryanc Mc Donald (Google inc., for  the sancl 2012 shared task)
// ===>  making it less sensitive to punct POS errors leading to
// mismatch of length


August 2013, 10
// Modified by DjamŽ Seddah (Univ. Paris Sorbonne, for the spmrl 2013 shared  task)
// ===> making it able to cope with Arabic very long lines (byte wise)
// ===>  now limit is 50000 bytes, was 5000 (tricky bug, if you ask me)
// please check the constant macro section if you encounter weird bugs not present in other
implementations (check evalC by Federico Sangatti for example, http://homepages.inf.ed.ac.uk/fsangati/evalC_25_5_10.zip )


August 2013, 23
// Modif from Thomas Müller (IMS Stuttgart)
// ===> adding of # in the stop word modify_label function (so that the
// lexer will read NPP instead of NPP##feat:...### as in hte SPMRL Data set
// Modif from Djamé Seddah
// ===>  Application of modify_label to all labels (including the POS label
// wich were left untouched for some reasons)
// That should btw be an option. (wether to evaluate full labels or not,
// only stripping of Non Terminal, POS tag and so on)
 

August 2013, 27
// Modif from Djamé
// --> adding of an option to include the non parsed sentences in the
// --> evaluation (-X option)
// --> adding an option to evaluate only the first N sentences (-K n)
// --> adding an option to provide a compact results view (-L) so one can do
// --> find ./ -name "*parsed.run?" -exec evalb_spmrl -L GOLD {} \; -print |
// --> grep -v '=====' | grep '='

September 2013, 6
// Modif from DJame
// fixing the infinite slowness bug (shame on me)
// now speed is similar to what it was before


October 2013, 13
// Addition from Djame
// Adding the spmrl_hebrew.prm if one wants to evaluate hebrew parsing within the
// same conditions as the state-of-the-art
// namely without counting the additional SYNpos layer which inflates evalb
// scores by almost 2 points.
// Note: for the spmrl shared task, we used the spmrl.prm file (so with
// these labels. It was too late to modify the rules once again when we
// realized this)

