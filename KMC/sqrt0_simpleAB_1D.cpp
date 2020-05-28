/*
 
 Title: Kinetic Monte Carlo Algorithm for one to one conversion.
 
 Author: Andres Garcia.
 
 Algorithm Type: Non-Rejection.
 
 Description: Kinetic Monte Carlo Algorithm for one to one diffusion conversion of molecules in a one-dimensional
 pore. The molecules can hop in and out of the pore, that is determined by the concentration of molecules outside
 of the fluid.
 
 Last Date Modified: October, 6 / 2016.
 
*/

//------------------------------------------------------
//   MODULES AND IMPORTS
//------------------------------------------------------

#include <algorithm>  // Has the functions for Min and Max (see documentation)
#include <fstream>  // To write the info in a file
#include <iostream>  // Basic input/output file
#include <math.h>  // Trigonometric, exponential, logarithmic functions, etc.
#include <random>  // Includes the random number generators (we want the Mersenne Twister, mt19937_64)
#include <sstream>  // Creates a special input/output stream
#include <string>  // To be able to use cout with strings and other basic operations
#include <time.h>  // Contains time functions to seed the generator
#include <vector>  // Use of dynamic arrays to select moves

using namespace std;

//------------------------------------------------------
//   RANDOM NUMBER GENERATOR(S)
//------------------------------------------------------

random_device rd;
unsigned int seed = rd();  // Generate a random number to seed the mt19937_64
mt19937_64 generator(seed);  // Get the random number generator
uniform_real_distribution<long double> rand_real3(0.0L, 1.0L);  // The distribution over which to generate numbers (non-inclusive range)

//------------------------------------------------------
//   FILE NAMES AND SAVE OPTIONS
//------------------------------------------------------

const string FL_NM_CUSTM = "simpAB-custm.csv";
const string FL_NM_DOUBL = "simpAB-doubl.csv";
const string FL_NM_DOUBR = "simpAB-doubR.csv";
const string FL_NM_QUART = "simpAB-quart.csv";
const string FL_NM_QUINT = "simpAB-quint.csv";
const string FL_NM_SINGL = "simpAB-singl.csv";
const string FL_NM_SINGL_SYM = "simpAB-singl-sym.csv";
const string FL_NM_TRIPL = "simpAB-tripl.csv";

const string FL_NM_POREC = "simpAB-poreConfig.csv";

const int OPEN_FL = 1;
const int WRIT_FL = 2;

//------------------------------------------------------
//   CONSTANTS
//------------------------------------------------------

// Length of the pore
const int PORELEN = 100;

// Number of atoms and identifiers for the system
const int AA = 0;
const int BB = 1;
const int EMPTY = 2;

const int NATOMS = 2;
const int DOUBL = 9;  // pow(3, 2)
const int TRIPL = 27;  // pow(3,3)
const int QUART = 81;  // pow(3, 4)
const int QUINT = 243;  // pow(3,5)

// Validation identifiers
const int MSWAP = 1;
const int MRXNS = 2;

// Update list identifiers
const int ADDMV = 1;
const int DELMV = 2;

// Number of moves available to the system are six: 2 hop (1 hop * 2 molecules), 2 end moves, 2 reaction moves, 1 swap move
const int NMOVES = 2 + 2 + 2 + 1;

// Equilibrium values 
const long double x_equil_A = 0.3L;
const long double x_equil_B = 0.47L;
const long double x_equil = x_equil_A + x_equil_B;

// Exchange probability
const long double pex = 0.82L;

// Hop rates for the molecules in the system
const long double hA_L = 1.0L;
const long double hB_L = 2.0L; 
const long double hE_AB = pex * 0.5L * (hA_L + hB_L);

// Reaction rate for the molecules
const long double rxn_A = 0.1L;  // Reaction rate for a molecule A to go to B
const long double rxn_B = 0.5L * rxn_A;  // Reaction rate for a molecule B to go to A

// Adsorption and desorption rates
const long double ads_rate = hA_L * x_equil_A + hB_L * x_equil_B;
const long double des_rate_A = hA_L * (1.0L - x_equil) + hE_AB * x_equil_B;
const long double des_rate_B = hB_L * (1.0L - x_equil) + hE_AB * x_equil_A;

// Time constants
const long double eqTime = 500000.0L;
const long double maxTime = 2.0L * eqTime;

//------------------------------------------------------
//   VARIABLES
//------------------------------------------------------

// Array that contains the atom types of the system
int atom_types[PORELEN];

// Array that contains the rates of the system
long double rates[NMOVES];

// Rate related parameters
long double total_system_rate;  // Total rate of the system

// Time variables
long double time_ell;  // Elapsed time
long double time_start;  // Time at which statistics have to be taken

// Variables to keep statistics
long double accum_stats_singl[PORELEN][NATOMS];
long double accum_stats_doubL[PORELEN][DOUBL];
long double accum_stats_doubR[PORELEN][DOUBL];
long double accum_stats_tripl[PORELEN][TRIPL];
long double accum_stats_quart[PORELEN][QUART];
long double accum_stats_quint[PORELEN][QUINT];

long double last_move_time_singl[PORELEN];
long double last_move_time_doubL[PORELEN];
long double last_move_time_doubR[PORELEN];
long double last_move_time_tripl[PORELEN];
long double last_move_time_quart[PORELEN];
long double last_move_time_quint[PORELEN];

bool track_stats;

// Arrays that contain the moves for the hops in the pore
int hop_mvs_A[PORELEN];
int hop_mvs_B[PORELEN];
int swp_mvs_AB[PORELEN];
int rxn_mvs_A[PORELEN];
int rxn_mvs_B[PORELEN];

// Pointers to the arrays that contain the moves for the hops in the pore
int hop_ptr_A[PORELEN];
int hop_ptr_B[PORELEN];
int swp_ptr_AB[PORELEN];
int rxn_ptr_A[PORELEN];
int rxn_ptr_B[PORELEN];

// The number of moves available in the pore
int n_mvs_A;
int n_mvs_B;
int n_mvs_rxn_A;
int n_mvs_rxn_B;
int n_mvs_swp_AB;

// Percentage progress
long double perc;

//------------------------------------------------------
//   SAVE AND LOAD FILE CONSTANTS AND VARIABLES
//------------------------------------------------------

//Name of the file where to save the simulation state
const string STATE_FILE = "state.csv";
const string GEN_STATE_FILE = "genState.txt";

// If the simulation should be saved/loaded variables
const bool LOAD_FILE = false;
const bool SAVE_FILE = true;

// How often the simulation should be saved
const long double saveInterval = 500000.0L;

// Variables
int saveCont;
long double saveTime;

//------------------------------------------------------
//   FUNCTION PROTOTYPES
//------------------------------------------------------

// Do move subroutines
void do_enm(int);
void do_rxn(int, int);
void do_swp(int, int, int);

// File save and load subroutines
void fr_load(string, string);
void fr_load_pr();
void fr_save(string, string);
void fr_save_pr();

// Get information subroutines
int get_num_rep(char);
long double get_stats_doubL(int, string);
long double get_stats_doubR(int, string);
long double get_stats_quart(int, string);
long double get_stats_quint(int, string);
long double get_stats_tripl(int, string);
string get_string_rep(int);
long double get_time_cont();

// Rate related subroutines
void rates_calculate();
void rates_select();

// Statistic related subroutines
void record_final_stats();
void record_stats_after_enm(int);
void record_stats_after_rxn(int);
void record_stats_after_swp(int);
void record_stats_doubL(int);
void record_stats_doubR(int);
void record_stats_quart(int);
void record_stats_quint(int);
void record_stats_singl(int);
void record_stats_tripl(int);

//Setup functions
void setup_pore();
void setup_system();
void setup_time_start(long double);

// List update subroutines
void update_lists(int, int [], int [], int *, int);
void update_lists_after_enm(int);
void update_lists_after_rxn(int);
void update_lists_after_swp(int);
void update_lists_rxn(int);
void update_lists_swp(int);

// Validation subroutines
void validate_move(int, int, int, int);
void validate_parameters();

// Write to file subroutines
void write_file_prelude(ofstream *);
void write_info_to_file();
void write_info_to_file_custm();
void write_info_to_file_doubL();
void write_info_to_file_doubR();
void write_info_to_file_quart();
void write_info_to_file_quint();
void write_info_to_file_singl();
void write_info_to_file_singl_sym();
void write_info_to_file_tripl();

// Other subroutines
int int_pow(int, int);
void print_pore(int);
void sim_percentage(long double);
vector <string> tokenize_string(string, char, int);

//------------------------------------------------------
//   MAIN FUNCTION
//------------------------------------------------------

/*
 
 Main program, runs the code until it's done.
 
*/
int main(){
	
	//Auxiliary variables
	long double dt;
	
	std::cout << "Non-Rejection: Simple A -> B." << endl;
	cout << "Program starts!" << endl;
	
	setup_system();
	
	do{
		
		//Calculate the rates
		rates_calculate();
		
		// Get the time advance
		dt = -log(rand_real3(generator)) / total_system_rate;
		time_ell += dt;
		
		if ((!track_stats) && time_ell > eqTime){
			
			//Start tracking the statistics
			track_stats = true;
			
			// Set the starting time
			time_start = time_ell - dt;
			
			// Set the initial times
			setup_time_start(time_start);
			
		}
		
		// Get the simulation percentage
		sim_percentage(time_ell);
		
		// Get out of the loop if the simulation is done
		if (time_ell > maxTime) break;
		
		// Select a molecule and make a move
		rates_select();
		
		// Saves the status of the simulation
		if(SAVE_FILE) fr_save_pr();
		
	} while (true);
	
	//Take the final statistics
	record_final_stats();
	
	// Write the statistics to the file
	write_info_to_file();
	
	cout << "Done running the program!" << endl;
	
	return 0;
	
}

//------------------------------------------------------
//   DO MOVE RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 A molecule is adsorbed or desorbed or swapped at the end.
 
 PARAMETERS:
 
	- int site : The end site at which the move is taking place.
 
 RETURN:
 
*/
void do_enm(int site){
	
	// Auxiliry variables
	int newType;
	long double randNum;
	
	// By default the initial type is empty
	newType = EMPTY;
	
	if (atom_types[site] != EMPTY && hE_AB > 0.0L){  //Attempt an exchange adsorption move 
		
		// If there is an adsorption by swap possibility test it
		if (atom_types[site] == AA){
			
			randNum = rand_real3(generator) * des_rate_A;
			if (randNum < hE_AB * x_equil_B) newType = BB;
			
		}
		else if (atom_types[site] == BB){
			
			randNum = rand_real3(generator) * des_rate_B;
			if (randNum < hE_AB * x_equil_A) newType = AA;
			
		}
		
	}
	else if (atom_types[site] == EMPTY){  // Adsorb a molecule
		
		// Choose a molecule to adsorb
		randNum = rand_real3(generator) * ads_rate;
		
		// Adsorb the molecule
		if (randNum < hA_L * x_equil_A){
			
			newType = AA;
			
		}
		else if (randNum < (hA_L * x_equil_A + hB_L * x_equil_B)){
			
			newType = BB;
			
		}
		
	}
	
	// Record the stats if needed
	if (track_stats) record_stats_after_enm(site);
	
	// Change the atom type at the end site
	atom_types[site] = newType;
	
	// Update the lists after the end move
	update_lists_after_enm(site);
	
}

/*
 
 A molecule reacts and changes type to A or B, depending on the type of molecule.
 
 PARAMETERS:
 
	- int site : Site where the reaction will take place.
	- int partType : The molecule type that is going to change.
 
 RETURN:
 
*/
void do_rxn(int site, int partType){
	
	// Validate the reaction move
	//validate_move(site, partType, partType, MRXNS);
	
	// Take the statistics if needed
	if(track_stats) record_stats_after_rxn(site);
	
	// Get the new kind of molecule
	if (partType == AA) atom_types[site] = BB; 
	else if(partType == BB) atom_types[site] = AA;
	
	// Update the necessary lists after the rxn
	update_lists_after_rxn(site);
	
}

/*
 
 Given the left most site, is swaps two molecules within the pore.
 
 PARAMETERS:
 
	- int site : The left most site where the swap move is going to happen, within the pore.
	- int parType1 : The type of the first molecule to be exchanged.
	- int parType2 : The type of the second molecule to be exchanged.
 
 RETURN:
 
*/
void do_swp(int site, int parType1, int parType2){
	
	// Auxiliary variables
	int type1, type2;
	
	// Validate the swap
	//validate_move(site, parType1, parType2, MSWAP);
	
	// Take statistics if relevant
	if (track_stats) record_stats_after_swp(site);
	
	// Get the molecule types
	type1 = atom_types[site];
	type2 = atom_types[site + 1];
	
	// Make the swap
	atom_types[site] = type2;
	atom_types[site + 1] = type1;
	
	// Update the necessary lists after the swap
	update_lists_after_swp(site);
	
}

//------------------------------------------------------
//  FILE SAVE AND LOAD RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Loads the file that has been saved.
 
 PARAMETERS:
 
	- string stString : Name of the file with the simulation data to load.
	- string geString : Name of the file with the random number generator data to load.
 
 RETURN:
 
*/
void fr_load(string stString, string geString){
	
	//Auxiliary variables
	int i, j;
	string line;
	ifstream fileS;
	vector <string> tokens;
	
	//-------------------------------------
	// LOAD THE RANDOM NUMBER GENERATOR
	//-------------------------------------
	
	//Open the generator state file and overwrite it, if it's the case.
	fileS.open(geString);
	
	// If the file is open, load the data
	if(fileS.is_open()){
		
		// Save the generator and the seed
		fileS >> generator;
		fileS >> seed;
		
		// Close the file
		fileS.close();
		
	}
	
	//-------------------------------------
	// LOAD THE RELEVANT DATA
	//-------------------------------------
	
	//Open the generator state file and overwrite it, if it's the case.
	fileS.open(stString);
	
	// If the file is open, load the data
	if(fileS.is_open()){
		
		// Read the save counter status
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		saveCont = stoi(tokens[0]);
		
		// Read the ellapsed time and starting time
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		// Read the tracking statistics state
		i = stoi(line);
		track_stats = false;
		if(i == 1) track_stats = true;
		
		// Read the ellapsed time and starting time
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		time_ell = stold(tokens[0]);
		time_start = stold(tokens[1]);
		
		// Read the atom types
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) atom_types[i] = stoi(tokens[i]);
		
		// No need to write the states if the stats aren't being tracked
		if(!track_stats){
			
			fileS.close();
			return;
			
		}
		else{
			
			cout << "Taking statistics!" << endl;
			
		}
		
		// Read the last move time for the single site statistics
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_singl[i] = stold(tokens[i]);
		
		// Read the last move time for the double site statistics towards the left
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_doubL[i] = stold(tokens[i]);
		
		// Read the last move time for the double site statistics towards the right
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_doubR[i] = stold(tokens[i]);
		
		// Read the last move time for the triple site statistics
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_tripl[i] = stold(tokens[i]);
		
		// Read the last move time for the quartet site statistics
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_quart[i] = stold(tokens[i]);
		
		// Read the last move time for the quartet site statistics
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) last_move_time_quint[i] = stold(tokens[i]);
		
		// Read the accumulated statistics for the single sites
		for(j = 0; j < NATOMS; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_singl[i][j] = stold(tokens[i]);
			
		}
		
		// Read the accumulated statistics for the double sites to the left
		for(j = 0; j < DOUBL; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_doubL[i][j] = stold(tokens[i]);
			
		}
		
		// Read the accumulated statistics for the double sites to the right
		for(j = 0; j < DOUBL; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_doubR[i][j] = stold(tokens[i]);
			
		}
		
		// Read the accumulated statistics for the triple quantities
		for(j = 0; j < TRIPL; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_tripl[i][j] = stold(tokens[i]);
			
		}
		
		// Read the accumulated statistics for the quartet quantities
		for(j = 0; j < QUART; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_quart[i][j] = stold(tokens[i]);
			
		}
		
		// Read the accumulated statistics for the quintet quantities
		for(j = 0; j < QUINT; j++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', PORELEN);
			
			for(i = 0; i < PORELEN; i++) accum_stats_quint[i][j] = stold(tokens[i]);
			
		}
		
		// Close the file
		fileS.close();
		
	}
	
}

/*
 
 Calls the specific functions to load the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void fr_load_pr(){
	
	ifstream fAux1, fAux2;
	
	try{
		
		cout << "Loading file!" << endl;
		
		fAux1.open(to_string(1) + STATE_FILE);
		fAux2.open(to_string(1) + GEN_STATE_FILE);
		if(!fAux1.good() || !fAux2.good()) throw 1;
		fAux1.close();
		fAux2.close();
		
		fr_load(to_string(1) + STATE_FILE, to_string(1) + GEN_STATE_FILE);
		
	}
	catch(...){
		
		try {
			
			cout << "File 1 couldn't be loaded, trying to load file 2." << endl;
			fAux1.open(to_string(2) + STATE_FILE);
			fAux2.open(to_string(2) + GEN_STATE_FILE);
			if(!fAux1.good() || !fAux2.good()) throw 1;
			fAux1.close();
			fAux2.close();
			
			fr_load(to_string(2) + STATE_FILE, to_string(2) + GEN_STATE_FILE);
			
		}
		catch(...){
			
			char a;
			cout << "No file could be loaded, the program will end." << endl;
			cout << endl << "Press any key and hit enter to continue: " << endl;
			std::cin >> a;
			
			// End the code
			std::exit(EXIT_FAILURE);         
		}
		
	}
	
}

/*
 
 Saves the state of the simulation variables to continue running.
 
 PARAMETERS:
 
	- string stString : Name of the file with the simulation data to save.
	- string geString : Name of the file with the random number generator data to save.
 
 RETURN:
 
*/
void fr_save(string stString, string geString){
	
	//Auxiliary variables
	int i, j;
	ofstream fileS;
	
	//Open the generator state file and overwrite it, if it's the case.
	fileS.open(geString);
	
	// Save the generator and the seed
	fileS << generator << endl;
	fileS << seed << endl;
	
	// Close the file
	fileS.close();
	
	// Open the file and overwrite it, if it's the case.
	fileS.open(stString);
	
	// Save the save counter
	fileS << saveCont << endl;
	
	// Save to see if the stats are being tracked
	fileS << track_stats << endl;
	
	// Save the time variables
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << time_ell << "," << time_start << endl;
	
	// Save the atom types of the pore at the moment
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<int>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << atom_types[i];
		
	}
	fileS << endl;
	
	// No need to write the states if the stats aren't being tracked
	if(!track_stats){
		
		fileS.close();
		return;
		
	}
	
	// Save the last move time for single site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_singl[i];
		
	}
	fileS << endl;
	
	// Save the last move time for double left site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_doubL[i];
		
	}
	fileS << endl;
	
	// Save the last move time for double right site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_doubR[i];
		
	}
	fileS << endl;
	
	// Save the last move time for triplet site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_tripl[i];
		
	}
	fileS << endl;
	
	// Save the last move time for quartet site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_quart[i];
		
	}
	fileS << endl;
	
	// Save the last move time for quintet site statistics
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << last_move_time_quint[i];
		
	}
	fileS << endl;
	
	//Save the accumulated single site statistics
	for(j = 0; j < NATOMS; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_singl[i][j];
			
		}
		fileS << endl;   
		
	}
	
	//Save the accumulated double left site statistics
	for(j = 0; j < DOUBL; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_doubL[i][j];
			
		}
		fileS << endl;   
		
	}
	
	//Save the accumulated double right site statistics
	for(j = 0; j < DOUBL; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_doubR[i][j];
			
		}
		fileS << endl;   
		
	}
	
	//Save the accumulated triple site statistics
	for(j = 0; j < TRIPL; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_tripl[i][j];
			
		}
		fileS << endl;   
		
	}
	
	//Save the accumulated quartet site statistics
	for(j = 0; j < QUART; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_quart[i][j];
			
		}
		fileS << endl;   
		
	}
	
	//Save the accumulated quintet site statistics
	for(j = 0; j < QUINT; j++){
		
		for(i = 0; i < PORELEN; i++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(i != 0) fileS << ",";
			fileS << accum_stats_quint[i][j];
			
		}
		fileS << endl;   
		
	}
	
	// Close the file
	fileS.close();
	
}

/*
 
 Calls the specific functions to save the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void fr_save_pr(){
	
	if(time_ell >= get_time_cont()){
		
		fr_save(to_string(1) + STATE_FILE, to_string(1) +  GEN_STATE_FILE);
		fr_save(to_string(2) + STATE_FILE, to_string(2) +  GEN_STATE_FILE);
		saveCont += 1;
		
	}
	
}

//------------------------------------------------------
//  GET INFORMATION RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Given a string with a single character, it returns the numerical representation.
 
 PARAMETERS:
 
	- char numToGet : The string representation and the name of a molecule. 
 
 RETURN:
 
	- int charRep : The numerical representation of the character.
 
*/
int get_num_rep(char numToGet){
	
	// Auxiliary variables
	int charRep;
	
	charRep = -1;
	
	// Check if the character is a valid character
	if(!(numToGet == 'A' || numToGet == 'B' || numToGet == 'E')){
		
		char a;
		
		cout << "The desired molecule numerical representation is not recognized." << endl;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		exit(EXIT_FAILURE);
		
	}
	
	// Get the required numerical representation of the character.
	if (numToGet == 'A'){
		
		charRep = AA;
		
	}
	else if (numToGet == 'B'){
		
		charRep = BB;
		
	}
	else if(numToGet == 'E'){
		
		charRep = EMPTY;
		
	}
	
	// Return the required numerical representation of the character.
	return charRep;
	
}

/*
 
 Gets the double site probability, towards the left, of a certain configuration.
 
 PARAMETERS:
 
	- int site : The site for which the correlation is requested.
	- string reqCorr : The type of correlation required. Must be in the format "XY", where X,Y ={A,B,E}.
 
 RETURN:
 
	- long double doubLCorr : The requested correlation. If the passed string is not valid, it returns zero.
 
*/
long double get_stats_doubL(int site, string reqCorr){
	
	// Auxiliary variables
	long double doubLCorr;
	int j, num1, num2;
	
	// Initiallize the variable
	doubLCorr = 0.0L;
	
	//Validate the string
	if (reqCorr.length() != 2 || !(site > 0 && site < PORELEN)) return doubLCorr;
	
	//Get the required characters from the string
	num1 = get_num_rep(reqCorr[0]);
	num2 = get_num_rep(reqCorr[1]);
	
	// Get the required index
	j = num1 * int_pow(3, 1) + num2;
	
	doubLCorr = accum_stats_doubL[site][j];
	
	// Return the string representation
	return doubLCorr;
	
}

/*
 
 Gets the double site probability, towards the right, of a certain configuration.
 
 PARAMETERS:
 
	- int site : The site for which the correlation is requested.
	- string reqCorr : The type of correlation required. Must be in the format "XY", where X,Y ={A,B,E}.
 
 RETURN:
 
	- long double doubRCorr : The requested correlation. If the passed string is not valid, it returns zero.
 
*/
long double get_stats_doubR(int site, string reqCorr){
	
	// Auxiliary variables
	long double doubRCorr;
	int j, num1, num2;
	
	// Initiallize the variable
	doubRCorr = 0.0L;
	
	//Validate the string
	if (reqCorr.length() != 2 || !(site >= 0 && site < (PORELEN-1))) return doubRCorr;
	
	//Get the required characters from the string
	num1 = get_num_rep(reqCorr[0]);
	num2 = get_num_rep(reqCorr[1]);
	
	// Get the required index
	j = num1 * int_pow(3, 1) + num2;
	
	doubRCorr = accum_stats_doubR[site][j];
	
	// Return the string representation
	return doubRCorr;
	
}

/*
 
 Gets the quartet site probability of a certain configuration.
 
 PARAMETERS:
 
	- int site : The site for which the correlation is requested.
	- string reqCorr : The type of correlation required. Must be in the format "WXYZ", where W,X,Y,Z ={A,B,E}.
 
 RETURN:
 
	- long double quartCorr : The requested correlation. If the passed string is not valid, it returns zero.
 
*/
long double get_stats_quart(int site, string reqCorr){
	
	// Auxiliary variables
	long double quartCorr;
	int j, num1, num2, num3, num4;
	
	// Initiallize the variable
	quartCorr = 0.0L;
	
	//Validate the string
	if (reqCorr.length() != 4 || !(site > 0 && site < (PORELEN - 2))) return quartCorr;
	
	//Get the required characters from the string
	num1 = get_num_rep(reqCorr[0]);
	num2 = get_num_rep(reqCorr[1]);
	num3 = get_num_rep(reqCorr[2]);
	num4 = get_num_rep(reqCorr[3]);
	
	// Get the required index
	j = num1 * int_pow(3, 3) + num2 * int_pow(3, 2) + num3 * int_pow(3, 1) + num4;
	
	quartCorr = accum_stats_quart[site][j];
	
	// Return the string representation
	return quartCorr;
	
}

/*
 
 Gets the quintet site probability of a certain configuration.
 
 PARAMETERS:
 
	- int site : The site for which the correlation is requested.
	- string reqCorr : The type of correlation required. Must be in the format "VWXYZ", where V,W,X,Y,Z ={A,B,E}.
 
 RETURN:
 
	- long double quintCorr : The requested correlation. If the passed string is not valid, it returns zero.
 
*/
long double get_stats_quint(int site, string reqCorr){
	
	// Auxiliary variables
	long double quintCorr;
	int j, num1, num2, num3, num4, num5;
	
	// Initiallize the variable
	quintCorr = 0.0L;
	
	//Validate the string
	if (reqCorr.length() != 5 || !(site > 1 && site < (PORELEN - 2))) return quintCorr;
	
	//Get the required characters from the string
	num1 = get_num_rep(reqCorr[0]);
	num2 = get_num_rep(reqCorr[1]);
	num3 = get_num_rep(reqCorr[2]);
	num4 = get_num_rep(reqCorr[3]);
	num5 = get_num_rep(reqCorr[4]);
	
	// Get the required index
	j = num1 * int_pow(3, 4) + num2 * int_pow(3, 3) + num3 * int_pow(3, 2) + num4 * int_pow(3, 1) + num5;
	
	quintCorr = accum_stats_quint[site][j];
	
	// Return the string representation
	return quintCorr;
	
}

/*
 
 Gets the triple site probability of a certain configuration.
 
 PARAMETERS:
 
	- int site : The site for which the correlation is requested.
	- string reqCorr : The type of correlation required. Must be in the format "XYZ", where X,Y,Z ={A,B,E}.
 
 RETURN:
 
	- long double triplCorr : The requested correlation. If the passed string is not valid, it returns zero.
 
*/
long double get_stats_tripl(int site, string reqCorr){
	
	// Auxiliary variables
	long double triplCorr;
	int j, num1, num2, num3;
	
	// Initiallize the variable
	triplCorr = 0.0L;
	
	//Validate the string
	if (reqCorr.length() != 3 || !(site > 0 && site < (PORELEN - 1))) return triplCorr;
	
	//Get the required characters from the string
	num1 = get_num_rep(reqCorr[0]);
	num2 = get_num_rep(reqCorr[1]);
	num3 = get_num_rep(reqCorr[2]);
	
	// Get the required index
	j = num1 * int_pow(3, 2) + num2 * int_pow(3, 1) + num3;
	
	triplCorr = accum_stats_tripl[site][j];
	
	// Return the string representation
	return triplCorr;
	
}

/*
 
 Given an integer that represents a molecule, it returns the string representation.
 
 PARAMETERS:
 
	- int stringToGet : The string representation and the name of a molecule.
 
 RETURN:
 
	- string numRep : The string representation of the integer.
 
*/
string get_string_rep(int stringToGet){
	
	// Auxiliary variables
	string numRep;
	
	numRep = "";
	
	// Check if the character is a valid character
	if (!(stringToGet == AA || stringToGet == BB || stringToGet == EMPTY)){
		
		char a;
		
		cout << "The desired string representation is not recognized." << endl;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		exit(EXIT_FAILURE);
		
	}
	
	// Get the desired string representation.
	if (stringToGet == AA){
		
		numRep = "A";
		
	}
	else if (stringToGet == BB){
		
		numRep = "B";
		
	}
	else if (stringToGet == EMPTY){
		
		numRep = "E";
		
	}
	
	// Return the string representation
	return numRep;
	
}

/*
 
 Gets the next time interval to save the file.
 
 PARAMETERS:
 
 RETURN:
 
	- long double niTime : The next time at which the file should be saved.
 
*/
long double get_time_cont(){
	
	// Auxiliary variables
	long double niTime;
	
	niTime = saveInterval * (long double) saveCont;
	
	return niTime;
	
}

//------------------------------------------------------
//   RATE RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Calculates the rates of the system.
 
 PARAMETERS:
 
 RETURN:
 
*/
void rates_calculate(){
	
	// Hop moves of A and B, and hop swap moves
	rates[0] = hA_L * ((long double) n_mvs_A);
	rates[1] = rates[0] + hB_L * ((long double) n_mvs_B);
	
	// Exchange rate
	rates[2] = rates[1] + hE_AB * ((long double) n_mvs_swp_AB);
	
	// End move at left end
	if(atom_types[0] == AA) rates[3] = rates[2] + des_rate_A;
	else if(atom_types[0] == BB) rates[3] = rates[2] + des_rate_B;
	else rates[3] = rates[2] + ads_rate;
	
	// End move at right end
	if(atom_types[PORELEN-1] == AA) rates[4] = rates[3] + des_rate_A;
	else if(atom_types[PORELEN-1] == BB) rates[4] = rates[3] + des_rate_B;
	else rates[4] = rates[3] + ads_rate;
	
	// Reaction moves of A and B
	rates[5] = rates[4] + rxn_A * ((long double) n_mvs_rxn_A);
	rates[6] = rates[5] + rxn_B * ((long double) n_mvs_rxn_B);
	
	// Get the total system rate
	total_system_rate = rates[NMOVES - 1];
	
}

/*
 
 Randomly selects a move based on the total rate of the system.
 
 PARAMETERS:
 
 RETURN:
 
*/
void rates_select(){
	
	// Auxiliary variables
	int site;
	long double randMv;
	
	// Choose a random number based on the total rate of the system
	randMv = rand_real3(generator) * rates[NMOVES-1];
	
	//Choose the move
	if (randMv < rates[0]){  // Move an A molecule to an adjacent empty site
		
		uniform_int_distribution <int> move_id(1, n_mvs_A);
		site = hop_mvs_A[move_id(generator) - 1];
		
		do_swp(site, AA, EMPTY);
		
	}
	else if (randMv < rates[1]){  // Move a B molecule to an adjacent empty site
		
		uniform_int_distribution <int> move_id(1, n_mvs_B);
		site = hop_mvs_B[move_id(generator) - 1];
		
		do_swp(site, BB, EMPTY);
		
	}
	else if (randMv < rates[2]){  // Swap and A  and B molecule on adjacent sites
		
		uniform_int_distribution <int> move_id(1, n_mvs_swp_AB);
		site = swp_mvs_AB[move_id(generator) - 1];
		
		do_swp(site, AA, BB);
		
	}
	else if (randMv < rates[3]){  // Do an end move on the left side
		
		do_enm(0);
		
	}
	else if (randMv < rates[4]){  // Do an end move on the right side
		
		do_enm(PORELEN-1);
		
	}
	else if (randMv < rates[5]){  // React an A molecule to make a B molecule
		
		uniform_int_distribution <int> move_id(1, n_mvs_rxn_A);
		site = rxn_mvs_A[move_id(generator) - 1];
		
		do_rxn(site, AA);
		
	}
	else if (randMv < rates[6]){  // React a B molecule to make an A molecule
		
		uniform_int_distribution <int> move_id(1, n_mvs_rxn_B);
		site = rxn_mvs_B[move_id(generator) - 1];
		
		do_rxn(site, BB);
		
	}
	else{
		
		char a;
		
		cout << "Error, the rate chosen is not valid: " << endl;
		cout << "\tTotal System Rate: " << total_system_rate << endl;
		cout << "\tRate Chosen: " << randMv << endl;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		
		// End the code
		exit(EXIT_FAILURE);
		
	}
	
}

//------------------------------------------------------
//   RECORD STATISTICS RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Records the final statistics and averages the statistics.
 
 PARAMETERS:
 
 RETURN:
 
*/
void record_final_stats(){
	
	//Auxiliary variables
	int i, j;
	long double dt;
	
	// Take the final statistics
	for (i = 0; i < PORELEN; i++){
		
		record_stats_singl(i);
		record_stats_doubL(i);
		record_stats_doubR(i);
		record_stats_tripl(i);
		record_stats_quart(i);
		record_stats_quint(i);
		
	}
	
	dt = (time_ell - time_start);
	
	// Check if the statistics can be averaged
	if ((time_ell - time_start) <= 0.0L){
		
		char a;
		
		cout << "The time difference to average is not suitable." << endl;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		
		// End the code
		exit(EXIT_FAILURE);
		
	}
	
	// Average the statistics
	for (i = 0; i < PORELEN; i++){
		
		for (j = 0; j < QUINT; j++){
			
			if(j < NATOMS) accum_stats_singl[i][j] = accum_stats_singl[i][j] / dt;
			if (j < DOUBL) {
				
				accum_stats_doubL[i][j] = accum_stats_doubL[i][j] / dt;
				accum_stats_doubR[i][j] = accum_stats_doubR[i][j] / dt;
				
			}
			if(j < TRIPL) accum_stats_tripl[i][j] = accum_stats_tripl[i][j] / dt;
			if(j < QUART) accum_stats_quart[i][j] = accum_stats_quart[i][j] / dt;
			if(j < QUINT) accum_stats_quint[i][j] = accum_stats_quint[i][j] / dt;
			
		}
		
	}
	
}

/*
 
 Records the needed statistics after an end move.
 
 PARAMETERS:
 
	- int site : The end site at which the move is taking place.
 
 RETURN:
 
*/
void record_stats_after_enm(int site){
	
	// Record single site statistics
	record_stats_singl(site);
	
	// Record double statistics to the left
	record_stats_doubL(site);
	record_stats_doubL(site + 1);
	
	// Record double statistics to the right
	record_stats_doubR(site - 1);
	record_stats_doubR(site);
	
	// Record triple statistics
	record_stats_tripl(site - 1);
	record_stats_tripl(site);
	record_stats_tripl(site + 1);
	
	// Record quartet statistics
	record_stats_quart(site - 2);
	record_stats_quart(site - 1);
	record_stats_quart(site);
	record_stats_quart(site + 1);
	
	// Record quintet statistics
	record_stats_quint(site - 2);
	record_stats_quint(site - 1);
	record_stats_quint(site);
	record_stats_quint(site + 1);
	record_stats_quint(site + 2);
	
}

/*
 
 Records the needed statistics after a reaction.
 
 PARAMETERS:
 
	- int site : The site where the reaction move is going to happen, within the pore.
 
 RETURN:
 
*/
void record_stats_after_rxn(int site){
	
	// Record single site statistics
	record_stats_singl(site);
	
	// Record double statistics to the left
	record_stats_doubL(site);
	record_stats_doubL(site + 1);
	
	// Record double statistics to the right
	record_stats_doubR(site - 1);
	record_stats_doubR(site);
	
	// Record triple statistics
	record_stats_tripl(site - 1);
	record_stats_tripl(site);
	record_stats_tripl(site + 1);
	
	// Record quartet statistics
	record_stats_quart(site - 2);
	record_stats_quart(site - 1);
	record_stats_quart(site);
	record_stats_quart(site + 1);
	
	// Record quintet statistics
	record_stats_quint(site - 2);
	record_stats_quint(site - 1);
	record_stats_quint(site);
	record_stats_quint(site + 1);
	record_stats_quint(site + 2);
	
}

/*
 
 Records the needed statistics after a swap.
 
 PARAMETERS:
 
	- int site : The left most site where the swap move is going to happen, within the pore.
 
 RETURN:
 
*/
void record_stats_after_swp(int site){
	
	// Record single site statistics
	record_stats_singl(site);
	record_stats_singl(site + 1);
	
	// Record double statistics to the left
	record_stats_doubL(site);
	record_stats_doubL(site + 1);
	record_stats_doubL(site + 2);
	
	// Record double statistics to the right
	record_stats_doubR(site - 1);
	record_stats_doubR(site);
	record_stats_doubR(site + 1);
	
	// Record triple statistics
	record_stats_tripl(site - 1);
	record_stats_tripl(site);
	record_stats_tripl(site + 1);
	record_stats_tripl(site + 2);
	
	// Record quartet statistics
	record_stats_quart(site - 2);
	record_stats_quart(site - 1);
	record_stats_quart(site);
	record_stats_quart(site + 1);
	record_stats_quart(site + 2);
	
	// Record quintet statistics
	record_stats_quint(site - 2);
	record_stats_quint(site - 1);
	record_stats_quint(site);
	record_stats_quint(site + 1);
	record_stats_quint(site + 2);
	record_stats_quint(site + 3);
	
}

/*
 
 Records the double statistics, towards the left, for a single site, of the type <Xn-1Yn>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_doubL(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type1, atom_type2, j;
	
	// If the site is not within the pore, ignore it
	if (!(site > 0 && site < PORELEN)) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_doubL[site];
	last_move_time_doubL[site] = time_ell;
	
	// Get the atom types for the sites
	atom_type1 = atom_types[site-1];
	atom_type2 = atom_types[site];
	
	// Get the correlation id
	j = atom_type1 * int_pow(3,1) + atom_type2;
	
	// Record the statistic
	accum_stats_doubL[site][j] += dt;
	
}

/*
 
 Records the double statistics, towards the right, for a single site, of the type <XnYn+1>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_doubR(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type1, atom_type2, j;
	
	// If the site is not within the pore, ignore it
	if (!(site >= 0 && site < (PORELEN-1))) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_doubR[site];
	last_move_time_doubR[site] = time_ell;
	
	// Get the atom types for the sites
	atom_type1 = atom_types[site];
	atom_type2 = atom_types[site+1];
	
	// Get the correlation id
	j = atom_type1 * int_pow(3, 1) + atom_type2;
	
	// Record the statistic
	accum_stats_doubR[site][j] += dt;
	
}

/*
 
 Records the quartet statistics, for a single site, of the type <Wn-1XnYn+1Zn+2>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_quart(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type1, atom_type2, atom_type3, atom_type4, j;
	
	// If the site is not within the pore, ignore it
	if (!(site > 0 && site < (PORELEN - 2))) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_quart[site];
	last_move_time_quart[site] = time_ell;
	
	// Get the atom types for the sites
	atom_type1 = atom_types[site - 1];
	atom_type2 = atom_types[site];
	atom_type3 = atom_types[site + 1];
	atom_type4 = atom_types[site + 2];
	
	// Get the correlation id
	j = atom_type1 * int_pow(3, 3) + atom_type2 * int_pow(3, 2) + atom_type3 * int_pow(3, 1) + atom_type4;
	
	// Record the statistic
	accum_stats_quart[site][j] += dt;
	
}

/*
 
 Records the quintet statistics, for a single site, of the type <Vn-2Wn-1XnYn+1Zn+2>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_quint(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type1, atom_type2, atom_type3, atom_type4, atom_type5, j;
	
	// If the site is not within the pore, ignore it
	if (!(site > 1 && site < (PORELEN - 2))) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_quint[site];
	last_move_time_quint[site] = time_ell;
	
	// Get the atom types for the sites
	atom_type1 = atom_types[site - 2];
	atom_type2 = atom_types[site- 1];
	atom_type3 = atom_types[site];
	atom_type4 = atom_types[site + 1];
	atom_type5 = atom_types[site + 2];
	
	// Get the correlation id
	j = atom_type1 * int_pow(3, 4) + atom_type2 * int_pow(3, 3) + atom_type3 * int_pow(3, 2) + atom_type4 * int_pow(3, 1) + atom_type5;
	
	// Record the statistic
	accum_stats_quint[site][j] += dt;
	
}

/*
 
 Records the statistics for a single site, of the type <Xn>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_singl(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type;
	
	// If the site is not within the pore, ignore it
	if (!(site >= 0 && site < PORELEN)) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_singl[site];
	last_move_time_singl[site] = time_ell;
	
	// Get the correlation id
	atom_type = atom_types[site];
	
	// Record the statistic
	if (atom_type == EMPTY) return;
	
	accum_stats_singl[site][atom_type] += dt;
	
}

/*
 
 Records the triplet statistics, for a single site, of the type <Xn-1YnZn+1>.
 
 PARAMETERS:
 
	- int site : The site for which the move took place.
 
 RETURN:
 
*/
void record_stats_tripl(int site){
	
	//Auxiliary variables
	double dt;
	int atom_type1, atom_type2, atom_type3, j;
	
	// If the site is not within the pore, ignore it
	if (!(site > 0 && site < (PORELEN - 1))) return;
	
	// Get the time advance and update the last move time for the site
	dt = time_ell - last_move_time_tripl[site];
	last_move_time_tripl[site] = time_ell;
	
	// Get the atom types for the sites
	atom_type1 = atom_types[site - 1];
	atom_type2 = atom_types[site];
	atom_type3 = atom_types[site + 1];
	
	// Get the correlation id
	j = atom_type1 * int_pow(3, 2) + atom_type2 * int_pow(3, 1) + atom_type3;
	
	// Record the statistic
	accum_stats_tripl[site][j] += dt;
	
}

//------------------------------------------------------
//   SETUP SYSTEM RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Setups the pore for a simulation, it is setup to empty by default.
 
 PARAMETERS:
 
 RETURN:
 
*/
void setup_pore(){
	
	// Auxiliary variables
	int i;
	
	// Set the pore to empty
	for (i = 0; i < PORELEN; i++) atom_types[i] = EMPTY;
	
	// Load the file if needed
	if(LOAD_FILE) fr_load_pr();
	
	//Update the lists
	for (i = 0; i < PORELEN; i++){
		
		update_lists_swp(i);
		update_lists_rxn(i);
		
	}
	
}

/*
 
 Setups the system for a simulation.
 
 PARAMETERS:
 
 RETURN:
 
*/
void setup_system(){
	
	// Auxiliary variables
	int i, j;
	
	// Set the saving counter to 1
	saveCont = 1;
	
	// Tracking statistics is false at the beginning of the simulation
	track_stats = false;
	
	// Time variables set to zero
	time_ell = 0.0L;
	time_start = 0.0L;
	
	// Set the number of moves to zero
	n_mvs_A = 0;
	n_mvs_B = 0;
	n_mvs_swp_AB = 0;
	n_mvs_rxn_A = 0;
	n_mvs_rxn_B = 0;
	
	// Setup the rates to zero initially
	for (i = 0; i < PORELEN; i++){
		
		// Set the available moves and pointers to their initial value 
		hop_mvs_A[i] = -1;
		hop_mvs_B[i] = -1;
		swp_mvs_AB[i] = -1;
		rxn_mvs_A[i] = -1;
		rxn_mvs_B[i] = -1;
		
		hop_ptr_A[i] = -1;
		hop_ptr_B[i] = -1;
		swp_ptr_AB[i] = -1;
		rxn_ptr_A[i] = -1;
		rxn_ptr_B[i] = -1;  
		
		// Set the rates to zero
		if(i < NMOVES) rates[i] = 0.0L;
		
		// Reset the last move time to zero
		last_move_time_singl[i] = 0.0L;
		last_move_time_doubL[i] = 0.0L;
		last_move_time_doubR[i] = 0.0L;
		last_move_time_tripl[i] = 0.0L;
		last_move_time_quart[i] = 0.0L;
		last_move_time_quint[i] = 0.0L;
		
		// Set the accumulated statistics to zero
		for (j = 0; j < QUINT; j++){
			
			if(j < NATOMS) accum_stats_singl[i][j] = 0.0L;
			if (j < DOUBL) {
				
				accum_stats_doubL[i][j] = 0.0L;
				accum_stats_doubR[i][j] = 0.0L;
			}
			if (j < TRIPL) accum_stats_tripl[i][j] = 0.0L;
			if (j < QUART) accum_stats_quart[i][j] = 0.0L;
			if (j < QUINT) accum_stats_quint[i][j] = 0.0L;
			
		}
		
	}
	
	// Setup the pore
	setup_pore();
	
	// Validate the parameters
	validate_parameters();
	
	// Initialize percentage variables
	perc = 0.0L;
	sim_percentage(-1.0L);
	
}

/*
 
 Sets the starting time to start taking statistics.
 
 PARAMETERS:
 
	- long dobuble ts : The starting time for the simulation.
 
 RETURN:
 
*/
void setup_time_start(long double ts){
	
	for(int i = 0; i < PORELEN; i++){
		
		last_move_time_singl[i] = ts;
		last_move_time_doubL[i] = ts;
		last_move_time_doubR[i] = ts;
		last_move_time_tripl[i] = ts;
		last_move_time_quart[i] = ts;
		last_move_time_quint[i] = ts;
		
		for(int j = 0; j < QUINT; j++){
			
			if(j < NATOMS) accum_stats_singl[i][j] = 0.0L;
			if(j < DOUBL){
				
				accum_stats_doubL[i][j] = 0.0L;
				accum_stats_doubR[i][j] = 0.0L;
				
			}
			if(j < TRIPL)  accum_stats_tripl[i][j] = 0.0L;
			if(j < QUART)  accum_stats_quart[i][j] = 0.0L;
			accum_stats_quint[i][j] = 0.0L;
			
		}
		
	}
	
}

//------------------------------------------------------
//   UPDATE LISTS RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Adds a move to a list or deletes a move from a list.
 
 PARAMETERS:
 
	- int site : The integer to be added to the vector.
	- int [] ptr_list : The list of pointers to be updated.
	- int [] mvs_list : The list of moves to be updated.
	- int * n_moves : The number of moves currently available.
	- int * n_moves : The number of moves currently available.
	- int oper : The intended operation, if to add or delete a move.
 
 RETURN:
 
*/
void update_lists(int site, int ptr_list[], int mvs_list[], int * n_moves, int oper){
	
	// Auxiliary variables
	int move_id;
	
	if(oper == ADDMV){
		
		// Modify the proper sites
		ptr_list[site] = (*n_moves);
		mvs_list[(*n_moves)] = site;
		
		// Add one to the number of available moves
		(*n_moves) = (*n_moves) + 1;
		
	}
	else if(oper == DELMV){
		
		// Get the move id for the site
		move_id = ptr_list[site];
		
		// Only worry to delete a site if the site is on the list or the number of moves is greater than zero
		if(!(*n_moves > 0 && move_id >= 0)) return;
		
		// Get the site at the very end of the array
		mvs_list[move_id] = mvs_list[(*n_moves) - 1];
		mvs_list[(*n_moves)-1] = - 1;
		
		ptr_list[mvs_list[move_id]] = move_id;
		ptr_list[site] = - 1;
		
		// Delete the site from the pointer list and reduce the number of moves by one
		(*n_moves) = (*n_moves) - 1;
		
	}
	
}

/*
 
 Updates the needed lists after an end move.
 
 PARAMETERS:
 
	- int site : The end site at which the move is took place.
 
 RETURN:
 
*/
void update_lists_after_enm(int site){
	
	// Update the swap lists
	update_lists_swp(site-1);
	update_lists_swp(site);
	
	// Update the reaction lists
	update_lists_rxn(site);
	
}

/*
 
 Updates the needed lists after a reaction.
 
 PARAMETERS:
 
	- int site : The left most site where the reaction move happened, within the pore.
 
 RETURN:
 
*/
void update_lists_after_rxn(int site){
	
	// Update the swap lists
	update_lists_swp(site - 1);
	update_lists_swp(site);
	
	// Update the reaction lists
	update_lists_rxn(site);
	
}

/*
 
 Updates the needed lists after a swap.
 
 PARAMETERS:
 
	- int site : The left most site where the swap move happened, within the pore.
 
 RETURN:
 
*/
void update_lists_after_swp(int site){
	
	// Update the swap lists
	update_lists_swp(site - 1);
	update_lists_swp(site + 1);
	
	// Update the reaction lists
	update_lists_rxn(site);
	update_lists_rxn(site + 1);
	
}

/*
 
 Updates the swap list for a specific site within the range for reaction.
 
 PARAMETERS:
 
	- int site : The end site at which the reaction took place.
 
 RETURN:
 
*/
void update_lists_rxn(int site){
	
	// If the site can't exchange within the pore don't do anything
	if(!(site >= 0 && site < PORELEN && (rxn_A > 0.0L || rxn_B > 0.0L))) return;
	
	// Delete the move from the lists
	if(n_mvs_rxn_A > 0) update_lists(site, rxn_ptr_A, rxn_mvs_A, &n_mvs_rxn_A, DELMV);
	if(n_mvs_rxn_B > 0) update_lists(site, rxn_ptr_B, rxn_mvs_B, &n_mvs_rxn_B, DELMV);
	
	// Add a move to the lists if needed
	if(atom_types[site] == AA && rxn_A > 0.0L){
		
		update_lists(site, rxn_ptr_A, rxn_mvs_A, &n_mvs_rxn_A, ADDMV);
		return;
		
	}
	else if(atom_types[site] == BB && rxn_B > 0.0L){
		
		update_lists(site, rxn_ptr_B, rxn_mvs_B, &n_mvs_rxn_B, ADDMV);
		return;
		
	}
	
}

/*
 
 Updates the swap list for a specific site within the range for swap.
 
 PARAMETERS:
 
	- int site : The end site at which the move is took place.
 
 RETURN:
 
*/
void update_lists_swp(int site){
	
	// Auxiliary variables
	int type1, type2;
	
	// If the site can't exchange within the pore don't do anything
	if (!(site >= 0 && site < PORELEN - 1)) return;
	
	// Delete the move from the lists
	if(n_mvs_A > 0) update_lists(site, hop_ptr_A, hop_mvs_A, &n_mvs_A, DELMV);
	if(n_mvs_B > 0) update_lists(site, hop_ptr_B, hop_mvs_B, &n_mvs_B, DELMV);
	
	if(hE_AB > 0.0L) update_lists(site, swp_ptr_AB, swp_mvs_AB, &n_mvs_swp_AB, DELMV);
	
	// Get the atom types at the sites
	type1 = atom_types[site];
	type2 = atom_types[site + 1];
	
	// Add a move to the lists if  needed
	if (((type1 == AA && type2 == EMPTY) || (type1 == EMPTY && type2 == AA)) && hA_L > 0.0L){
		
		update_lists(site, hop_ptr_A, hop_mvs_A, &n_mvs_A, ADDMV);
		return;
		
	}
	else if (((type1 == BB && type2 == EMPTY) || (type1 == EMPTY && type2 == BB)) && hB_L > 0.0L){
		
		update_lists(site, hop_ptr_B, hop_mvs_B, &n_mvs_B, ADDMV);
		return;
		
	}
	else if(((type1 == AA && type2 == BB) || (type1 == BB && type2 == AA)) && hE_AB > 0.0L){
		
		update_lists(site, swp_ptr_AB, swp_mvs_AB, &n_mvs_swp_AB, ADDMV);
		return;
		
	}
	
}

//------------------------------------------------------
//   VALIDATION RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Validates if a move is allowed, given the needed parameters.
 
 PARAMETERS:
 
	- int site : The site within the pore
	- int part1 : The first kind of molecule of the move.
	- int part2 : The second kind of molecule of the move.
	- int mType : The kind of move.
 
 RETURN:
 
*/
void validate_move(int site, int part1, int part2, int mType){
	
	// Auxiliary variables
	int type1, type2;
	
	if(mType == MSWAP){
		
		//Check that the molecules are valid to swap
		if (site < 0 || site >= (PORELEN - 1)){
			
			char a;
			
			cout << "Not a valid site." << endl;
			cout << "The site should be in the range: [" << 0 << "," << (PORELEN-2) << "]"  << endl;
			cout << "site = " << site << endl;
			
			cout << endl << "Press any key and hit enter to continue: " << endl;
			cin >> a;
			
			// End the code
			exit(EXIT_FAILURE);
			
		}
		
		type1 = atom_types[site];
		type2 = atom_types[site + 1];
		
		//Check that the molecules are valid to swap
		if (!((type1 == part1 && type2 == part2) || (type1 == part2 && type2 == part1))){
			
			char a;
			
			cout << "The molecules that are being swapped are of the same kind, or not valid: " << endl;
			cout << "Particles to be swapped:\n\tType1: " << part1 << ", parType2: " << part2 << endl;
			cout << "site 1 = " << site << ", Particle type: " << type1 << endl;
			cout << "site 2 = " << (site + 1) << ", Particle type: " << type2 << endl;
			
			cout << endl << "Press any key and hit enter to continue: " << endl;
			cin >> a;
			
			// End the code
			exit(EXIT_FAILURE);
			
		}
		
		return;
		
	}
	else if(mType == MRXNS){
		
		//Check that the molecules are valid to swap
		if (site < 0 || site >= PORELEN){
			
			char a;
			
			cout << "Not a valid site." << endl;
			cout << "The site should be in the range: [" << 0 << "," << (PORELEN - 1) << "]"  << endl;
			cout << "site = " << site << endl;
			
			cout << endl << "Press any key and hit enter to continue: " << endl;
			cin >> a;
			
			// End the code
			exit(EXIT_FAILURE);
			
		}
		
		type1 = atom_types[site];
		
		//Check that the molecules are valid to swap
		if (type1 != part1){
			
			char a;
			
			cout << "The molecule that is reacting are not valid: " << endl;
			cout << "Particle to react:\n\tType1: " << part1 << endl;
			cout << "site 1 = " << site << ", Particle type: " << type1 << endl;
			
			cout << endl << "Press any key and hit enter to continue: " << endl;
			cin >> a;
			
			// End the code
			exit(EXIT_FAILURE);
			
		}
		
		return;
		
	}
	else{
		
		char a;
		
		cout << "Not a valid state to check." << endl;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		
		// End the code
		exit(EXIT_FAILURE);
		
	}
	
}

/*
 
 Validates that the parameters are appropriate for the simulation.
 
 PARAMETERS:
 
 RETURN:
 
*/
void validate_parameters(){
	
	// Auxiliary variables
	bool cond, cond1;
	
	cond = false;
	
	// Verify the pore length
	cond = (PORELEN < 4);
	cond1 = cond;
	if (cond) cout << "The pore is not long enough for a simulation to happen." << endl;
	
	// Verify the outside equilibrium conditions
	cond1 = (x_equil_A < 0.0L || x_equil_A >= 1.0L);
	cond1 = cond1 || (x_equil_B < 0.0L || x_equil_B >= 1.0L);
	cond1 = cond1 || (x_equil <= 0.0L || x_equil >= 1.0L);
	cond = cond || cond1;
	if (cond1) {
		
		cout << "The equilibrium conditions outside the pore are wrong:" << endl;
		cout << "\tThe total equilibrium concentration should be greater than 0.0 and less than 1.0." << endl;
		cout << "\tThe equilibrium concentration of A or B should be greater than or equal to 0.0 and less than 1.0." << endl;
		cout << "x_equil_A = " << x_equil_A << endl;
		cout << "x_equil_B = " << x_equil_B << endl;
		cout << "x_equil = " << x_equil << endl;
		
	}
	
	// Verify the hop rates
	cond1 = ((hA_L < 0.0L) || (hB_L < 0.0L) || (hE_AB < 0.0L));
	cond = cond || cond1;
	if (cond1) {
		
		cout << "One or more hop rates are wrong, hop_rate_i must be greater or equal to 0.0." << endl;
		cout << "Hop rate A = " << hA_L << endl;
		cout << "Hop rate B = " << hB_L << endl;
		cout << "Hop rate Ex = " << hE_AB << endl;
		
	}
	
	// Verify exchange probability
	cond1 = (pex < 0.0L || pex > 1.0L);
	cond = cond || cond1;
	if (cond1) {
		
		cout << "The exchange probability has to be greater than 0.0 and less than 1." << endl;
		cout << "pex = " << pex << endl;
		
	}
	
	// Verify the reaction constants
	cond1 = (rxn_A < 0.0L || rxn_B < 0.0L);
	cond = cond || cond1;
	if (cond1) {
		
		cout << "The reaction constants should be greater than or equal to zero." << endl;
		cout << "rxn_A = " << rxn_A << endl;
		cout << "rxn_B = " << rxn_B << endl;
		
	}
	
	// Exit the program
	if (cond){ 
		
		char a;
		
		cout << endl << "Press any key and hit enter to continue: " << endl;
		cin >> a;
		exit(EXIT_FAILURE);
		
	}
	
}

//------------------------------------------------------
//   WRITE FILES RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Writes the important simulation information to the an already opened file.
 
 PARAMETERS:
 
	- ofstream * fileName : The pointer to the open file to write the prelude.
 
 RETURN:
 
*/
void write_file_prelude(ofstream * fileName){
	
	// Model and algorithm type
	(*fileName) << "Non-Rejection SimpleAB";
	
	// Outside concentrations
	(*fileName) << ",<A>_out = " << x_equil_A;
	(*fileName) << ",<B>_out = " << x_equil_B;
	
	// Hop rates
	(*fileName) << ",h_a = " << hA_L;
	(*fileName) << ",h_b = " << hB_L;
	
	// Exchange rates
	(*fileName) << ",p_ex = " << pex ;
	(*fileName) << ",h_ex = " << hE_AB;
	
	// Reaction rates
	(*fileName) << ",K_ab = " << rxn_A;
	(*fileName) << ",K_ba = " << rxn_B;
	
	// Other information
	(*fileName) << ",Time_eq = "  << eqTime;
	(*fileName) << ",Time_max = "  << maxTime;
	
	//End the line
	(*fileName) << endl;
	
}

/*
 
 Writes the different required statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file(){
	
	// Write custom statistics
	write_info_to_file_custm();
	
	// Write the specific statistics
	write_info_to_file_singl();
	write_info_to_file_doubL();
	write_info_to_file_doubR();
	write_info_to_file_tripl();
	write_info_to_file_quart();
	write_info_to_file_quint();
	
	// Write the symmetrized statistics
	write_info_to_file_singl_sym();
	
}

/*
 
 Writes customized statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_custm(){
	
	//Auxiliary variables
	ofstream myFile;
	
	myFile.open(FL_NM_CUSTM);
	write_file_prelude(&myFile);
	
	myFile.close();
	
}

/*
 
 Writes the double statistics, to the left, in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_doubL(){
	
	//Auxiliary variables
	int i, j;
	string aux;
	ofstream myFile;
	long double avgs_quant;
	
	myFile.open(FL_NM_DOUBL);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "n";
	for (i = 0; i <= NATOMS; i++){
		
		for (j = 0; j <= NATOMS; j++){
			
			aux = ",<" + get_string_rep(i) + "n-1" + get_string_rep(j) + "n>";
			myFile << aux;
			
		}
		
	}
	myFile << endl;
	
	// Write the statistics
	for (i = 1; i < PORELEN; i++){
		
		myFile << (i + 1);
		
		for (j = 0; j < DOUBL; j++){
			
			// Get the statistics and write them
			avgs_quant = accum_stats_doubL[i][j];
			
			myFile.precision(15);
			myFile << "," << avgs_quant;
			
		}
		
		myFile << endl;
		
	}
	
	myFile.close();
	
}

/*
 
 Writes the double statistics, to the right, in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_doubR(){
	
	//Auxiliary variables
	int i, j;
	string aux;
	ofstream myFile;
	long double avgs_quant;
	
	myFile.open(FL_NM_DOUBR);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "n";
	for (i = 0; i <= NATOMS; i++){
		
		for (j = 0; j <= NATOMS; j++){
			
			aux = ",<" + get_string_rep(i) + "n" + get_string_rep(j) + "n+1>";
			myFile << aux;
			
		}
		
	}
	myFile << endl;
	
	// Write the statistics
	for (i = 0; i < PORELEN-1; i++){
		
		myFile << (i + 1);
		
		for (j = 0; j < DOUBL; j++){
			
			// Get the statistics and write them
			avgs_quant = accum_stats_doubR[i][j];
			
			myFile.precision(15);
			myFile << "," << avgs_quant;
			
		}
		
		myFile << endl;
		
	}
	
	myFile.close();
	
}


/*
 
 Writes the quartet statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_quart(){
	
	//Auxiliary variables
	int i, j, k, l;
	string aux;
	ofstream myFile;
	long double avgs_quant;
	
	myFile.open(FL_NM_QUART);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "n";
	for (i = 0; i <= NATOMS; i++){
		
		for (j = 0; j <= NATOMS; j++){
			
			for (k = 0; k <= NATOMS; k++){
				
				for (l = 0; l <= NATOMS; l++){
					
					aux = ",<" + get_string_rep(i) + "n-1" + get_string_rep(j) + "n" + get_string_rep(k) + "n+1" + get_string_rep(l) + "n+2>";
					myFile << aux;
					
				}
				
			}
			
		}
		
	}
	myFile << endl;
	
	// Write the statistics
	for (i = 1; i < PORELEN-2; i++){
		
		myFile << (i + 1);
		
		for (j = 0; j < QUART; j++){
			
			// Get the statistics and write them
			avgs_quant = accum_stats_quart[i][j];
			
			myFile.precision(15);
			myFile << "," << avgs_quant;
			
		}
		
		myFile << endl;
		
	}
	
	myFile.close();
	
}

/*
 
 Writes the quintet statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_quint(){
	
	//Auxiliary variables
	int i, j, k, l, m;
	string aux;
	ofstream myFile;
	long double avgs_quant;
	
	myFile.open(FL_NM_QUINT);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "n";
	for (i = 0; i <= NATOMS; i++){
		
		for (j = 0; j <= NATOMS; j++){
			
			for (k = 0; k <= NATOMS; k++){
				
				for (l = 0; l <= NATOMS; l++){
					
					for(m = 0; m <= NATOMS; m++){
						
						aux = ",<" + get_string_rep(i) + "n-2" + get_string_rep(j) + "n-1" + get_string_rep(k) + "n" + get_string_rep(l) + "n+1" + get_string_rep(m) + "n+2>";
						myFile << aux;
						
					}
					
				}
				
			}
			
		}
		
	}
	myFile << endl;
	
	// Write the statistics
	for (i = 2; i < PORELEN-2; i++){
		
		myFile << (i + 1);
		
		for (j = 0; j < QUINT; j++){
			
			// Get the statistics and write them
			avgs_quant = accum_stats_quint[i][j];
			
			myFile.precision(15);
			myFile << "," << avgs_quant;
			
		}
		
		myFile << endl;
		
	}
	
	myFile.close();
	
}

/*
 
 Writes the single statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_singl(){
	
	//Auxiliary variables
	ofstream myFile;
	long double avgs_A, avgs_B, avgs_E, avgs_X;
	
	myFile.open(FL_NM_SINGL);
	write_file_prelude(&myFile);
	
	myFile << "L,W,H,<A_LWH>,<B_LWH>,<X_LWH>=<A_LWH>+<B_LWH>,<E_LWH>" << endl;
	
	for (int i = 0; i < PORELEN; i++){
		
		avgs_A = accum_stats_singl[i][AA];
		avgs_B = accum_stats_singl[i][BB];
		avgs_X = avgs_A + avgs_B;
		avgs_E = 1.0L - avgs_X;
		
		myFile.precision(15);
		myFile << (i+1) << "," << 1 << "," << 1 << "," << avgs_A << "," << avgs_B << "," << avgs_A + avgs_B << "," << avgs_E << endl;
		
	}
	
	myFile.close();
	
}

/*
 
 Writes the symmetrized single statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_singl_sym(){
	
	//Auxiliary variables
	ofstream myFile;
	long double avgs_A, avgs_B, avgs_E, avgs_X;
	
	myFile.open(FL_NM_SINGL_SYM);
	write_file_prelude(&myFile);
	
	myFile << "L,W,H,<A_LWH>,<B_LWH>,<X_LWH>=<A_LWH>+<B_LWH>,<E_LWH>" << endl;
	
	for (int i = 0; i < PORELEN; i++){
		
		avgs_A = 0.5L * (accum_stats_singl[i][AA] + accum_stats_singl[PORELEN - 1 - i][AA]);
		avgs_B = 0.5L * (accum_stats_singl[i][BB] + accum_stats_singl[PORELEN - 1 - i][BB]);
		avgs_X = avgs_A + avgs_B;
		avgs_E = 1.0L - avgs_X;
		
		myFile.precision(15);
		myFile << (i+1) << "," << 1 << "," << 1 << "," << avgs_A << "," << avgs_B << "," << avgs_A + avgs_B << "," << avgs_E << endl;
		
	}
	
	myFile.close();
	
}

/*
 
 Writes the triple statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_tripl(){
	
	//Auxiliary variables
	int i, j, k;
	string aux;
	ofstream myFile;
	long double avgs_quant;
	
	myFile.open(FL_NM_TRIPL);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "n";
	for (i = 0; i <= NATOMS; i++){
		
		for (j = 0; j <= NATOMS; j++){
			
			for (k = 0; k <= NATOMS; k++){
				
				aux = ",<" + get_string_rep(i) + "n-1" + get_string_rep(j) + "n" + get_string_rep(k) +"n+1>";
				myFile << aux;
				
			}
			
		}
		
	}
	myFile << endl;
	
	// Write the statistics
	for (i = 1; i < PORELEN-1; i++){
		
		myFile << (i + 1);
		
		for (j = 0; j < TRIPL; j++){
			
			// Get the statistics and write them
			avgs_quant = accum_stats_tripl[i][j];
			
			myFile.precision(15);
			myFile << "," << avgs_quant;
			
		}
		
		myFile << endl;
		
	}
	
	myFile.close();
	
}

//------------------------------------------------------
// OTHER SUBROUTINES
//------------------------------------------------------

/*
 
 Gives the result of an integer raised to a positive integer, including zero. If the power is less than zero, it will return 1.
 
 PARAMETERS:
 
	- int num1 : The number to be raised to a certain power.
	- int num2 : The power to raise the first number.
 
 RETURN:
 
	- int powerR : The result of the operation.
 
*/
int int_pow(int num1, int num2){
	
	// Auxiliary variables
	int i, powerR;
	
	if (num2 <= 0){
		
		powerR = 1;
		
	}
	else if (num1 == 0){
		
		powerR = 0;
		
	}
	else{
		
		powerR = num1;
		
		for (i = 1; i < num2; i++) powerR = powerR *num1;
		
	}
	
	// Return the value
	return powerR;
	
}

/*
 
 Prints the pore configuration in a suitable format to be turned into an image.
 
 PARAMETERS:
 
	- int opt : The option of the information to be printed.
 
 RETURN:
 
*/
void print_pore(int opt){
	
	// Auxiliary variables
	int i, j;
	ofstream fileS;
	
	if(opt == OPEN_FL){
		
		// Create the file and over write it
		fileS.open(FL_NM_POREC);
		
		// Write the basic information to the file
		for(j = 0; j < 3; j++){
			
			for(i = 0; i < PORELEN; i++){
				
				if(i == 0 && j == 0) fileS << "L";
				else if(i == 0 && j == 1) fileS << "W";
				else if(i == 0 && j == 2) fileS << "H";
				
				if(j == 0) fileS  << "," << (i+1);
				if(j != 0) fileS << "," << 1;
				
			} 
			fileS << endl;
			
		}
		
		// Close the file
		fileS.close();
		
	}
	else if(opt == WRIT_FL){
		
		// Open the file to write the configuration 
		fileS.open(FL_NM_POREC, std::ofstream::out | std::ofstream::app);
		
		for(i = 0; i < PORELEN; i++){
			
			if(i == 0) fileS << "Atom Types";
			fileS << "," << ((atom_types[i] + 1) % (NATOMS + 1));
			
		}
		fileS << endl;
		
		// Close the file
		fileS.close();
		
	}
	
}

/*
 
 Prints the simulation percentage progress.
 
 PARAMETERS:
 
	- long double timeEllapsed : The time ellapsed.
 
 RETURN:
 
*/
void sim_percentage(long double timeEllapsed){
	
	// Auxiliary variables
	int i;
	long double auxTime;
	
	if(timeEllapsed < 0.0L){
		
		cout << "\t";
		for(i = 1; i <= 50; i++) cout << "x";
		for(i = 1; i <= 50; i++) cout << "o";
		cout << endl << "\t";
		cout.flush();
		
	}
	else{
		
		auxTime = 100.0L * (timeEllapsed / maxTime);
		
		while(auxTime > perc){ 
			
			perc += 1.0L;
			if(perc > 100.0L){
				
				cout << endl; 
				break;
				
			}
			if(perc <= 50.0L) cout << "x";
			else if(perc > 50.0L) cout << "o";
			cout.flush();
			
		}
		
	}
	
}

/*
 
 Tokenizes a given string into a vector, given a character. It can throw an exception.
 
 PARAMETERS:
 
	- string st : The string to be tokenized.
	- char tk : The character to be used as the token.
	- int len : The expected length of the array. If it is less than zero, the exception is ignored.
 
 RETURN:
 
	- vector <string> cont : The vector container with the tokens.
 
*/
vector <string> tokenize_string(string st, char tk, int len){
	
	// Auxiliary variables
	vector <string> cont;
	istringstream split(st);
	
	// Resize the vector to empty
	cont.resize(0);
	
	// Get the tokens in the list
	for(string line; getline(split, line, tk); cont.push_back(line));
	
	// Check if the length of the string is valid, otherwise throw an exception
	if(len >= 0){
		
		if((int) cont.size() != len) throw 1;
		
	}
	
	// Return the container with the tokens
	return cont;
	
}
