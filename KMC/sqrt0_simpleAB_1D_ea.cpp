/*
 
 Title: Kinetic Monte Carlo Algorithm for one to one conversion.
 
 Author: Andres Garcia.
 
 Algorithm Type: Non-Rejection.
 
 Descriptiton: Kinetic Monte Carlo Algorithm for one to one diffusion conversion of particles in a one dimensional
 pore. The particles can hop in and out of the pore, that is determined by the concentration of particles outside
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
#include <random>  // Includes the random number generators (we want the Mersene Twister, mt19937_64)
#include <sstream>  // Creates a special input/output stream
#include <string>  // To be able to use cout with strings and other basic operations
#include <time.h>  // Contains time functions, to seed the generator
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

const string FL_NM_GAMMA = "simpAB-gamma.csv";
const string FL_NM_SINGL = "simpAB-singl.csv";
const string FL_NM_SINGL_SYM = "simpAB-singl-sym.csv";

const string FL_NM_POREC = "simpAB-poreConfig.csv";

const int OPEN_FL = 1;
const int WRIT_FL = 2;

//------------------------------------------------------
//   CONSTANTS
//------------------------------------------------------

// Taking statistics variables
const bool INIT_SIM = true;
const bool RECO_STA = true;

// Number of simulations over which to average
const int NSIMS = 1000;

// Length of the pore
const int PORELEN = 100;

// Number of atoms and identifiers for the system
const int AA = 0;
const int BB = 1;
const int EMPTY = 2;
const int NATOMS = 2;

// Validation identifiers
const int MSWAP = 1;
const int MRXNS = 2;

// Update list identifiers
const int ADDMV = 1;
const int DELMV = 2;

// Number of moves available to the system are six: 2 hop (1 hop * 2 particles), 2 end moves, 2 reaction moves, 1 swap move
const int NMOVES = 2 + 2 + 2 + 1;

// Equilibrium values 
const long double x_equil_A = 0.3L;
const long double x_equil_B = 0.47L;
const long double x_equil = x_equil_A + x_equil_B;

// Exchange probability
const long double pex = 0.82L;

// Hop rates for the particles in the system
const long double hA_L = 1.0L;
const long double hB_L = 2.0L;
const long double hE_AB = pex * 0.5L * (hA_L + hB_L);

// Reaction rate for the particles
const long double rxn_A = 0.1L;  // Reaction rate for a particle A to go to B
const long double rxn_B = 0.5L * rxn_A;  // Reaction rate for a particle B to go to A

// Adsorption and desorption rates
const long double des_rate_A = hA_L * (1.0L - x_equil) + hE_AB * x_equil_B;
const long double des_rate_B = hB_L * (1.0L - x_equil) + hE_AB * x_equil_A;
const long double ads_rate = hA_L * x_equil_A + hB_L * x_equil_B;

// Time constants
const int MTSIZE = 5;
const long double maxTime[MTSIZE] = {0.0L, 5.0L, 25.0L, 125.0L, 625.0L};

//------------------------------------------------------
//   VARIABLES
//------------------------------------------------------

// Percentage of simulation complete
long double perc;

// Current simulation number
int nSims;
int nsSims;

// Array that contains the atom types of the system
int atom_types[PORELEN];

// Array that contains the rates of the system
long double rates[NMOVES];

// Rate related parameters
long double total_system_rate;  // Total rate of the system

// Time variables
long double time_ell;  // Elapsed time
vector <long double> timeTrack;

// Variables to keep statistics
long double accum_stats_singl[MTSIZE][PORELEN][NATOMS];
long double gammat[MTSIZE][NATOMS];

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

//------------------------------------------------------
//   SAVE AND LOAD FILE CONSTANTS AND VARIABLES
//------------------------------------------------------

//Name of the file where to save the simulation state
const string STATE_FILE = "state.csv";
const string GEN_STATE_FILE = "genState.txt";

const bool LOAD_FILE = false;
const bool SAVE_FILE = true;

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

// Rate related subroutines
void rates_calculate();
void rates_select();

// Statistic related subroutines
void record_stats(bool);
void record_stats_singl(bool, int);

//Setup functions
void setup_init_vals();
void setup_pore(bool);
void setup_system(bool);

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
void write_info_to_file_gamma();
void write_info_to_file_singl();
void write_info_to_file_singl_sym();

// Other subroutines
int int_pow(int, int);
void print_pore(int);
void sim_percentage(int);
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
	
	setup_system(INIT_SIM);
	
	cout << "\tRunning the simulation!" << endl;
	
	// Get the statistics
	for(nSims = nsSims; nSims <= NSIMS; nSims++){
		
		if(nSims == 1) sim_percentage(0);
		
		setup_system(!INIT_SIM);
		
		do{
			
			//Calculate the rates
			rates_calculate();
			
			// Get the time advance
			dt = -log(rand_real3(generator)) / total_system_rate;
			time_ell += dt;
			
			// Get out of the loop if the simulation is done
			if (time_ell >= timeTrack[0]) record_stats(RECO_STA);
			
			// Select a particle and make a move
			rates_select();
			
		}while (timeTrack.size() > 0);
		
		sim_percentage(nSims);
		
	}
	
	// Average the final statistics
	record_stats(!RECO_STA);
	
	// Write the statistics to the file
	write_info_to_file();
	
	cout << endl << "Done running the program!" << endl;
	
	return 0;
	
}

//------------------------------------------------------
//   DO MOVE RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 A particle is adsorbed or desorbed or swapped at the end.
 
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
	else if (atom_types[site] == EMPTY){  // Adsorb a particle
		
		// Choose a particle to adsorb
		randNum = rand_real3(generator) * ads_rate;
		
		// Adsorb the particle
		if (randNum < hA_L * x_equil_A){
			
			newType = AA;
			
		}
		else if (randNum < (hA_L * x_equil_A + hB_L * x_equil_B)){
			
			newType = BB;
			
		}
		
	}
	
	// Change the atom type at the end site
	atom_types[site] = newType;
	
	// Update the lists after the end move
	update_lists_after_enm(site);
	
}

/*
 
 A particle reacts and changes type to A or B, depending on the type of particle.
 
 PARAMETERS:
 
	- int site : Site where the reaction will take place.
	- int partType : The particle type that is going to change.
 
 RETURN:
 
*/
void do_rxn(int site, int partType){
	
	// Validate the reaction move
	//validate_move(site, partType, partType, MRXNS);
	
	// Get the new kind of particle
	if (partType == AA) atom_types[site] = BB; 
	else if(partType == BB) atom_types[site] = AA;
	
	// Update the necessary lists after the rxn
	update_lists_after_rxn(site);
	
}

/*
 
 Given the left most site, is swaps two particles within the pore.
 
 PARAMETERS:
 
	- int site : The left most site where the swap move is going to happen, within the pore.
	- int parType1 : The type of the first particle to be exchanged.
	- int parType2 : The type of the second particle to be exchanged.
 
 RETURN:
 
*/
void do_swp(int site, int parType1, int parType2){
	
	// Auxiliary variables
	int type1, type2;
	
	// Validate the swap
	//validate_move(site, parType1, parType2, MSWAP);
	
	// Get the particle types
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
	int i, j, k;
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
		
		// Read the ellapsed time and starting time
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		time_ell = stold(tokens[0]);
		nsSims = stoi(tokens[1]);
		
		// Read the atom types
		getline(fileS, line);
		tokens = tokenize_string(line, ',', PORELEN);
		
		for(i = 0; i < PORELEN; i++) atom_types[i] = stoi(tokens[i]);
		
		// Read the accumulated statistics for the single sites
		for(i = 0; i < MTSIZE; i++){
			
			for(k = 0; k < NATOMS; k++){
				
				getline(fileS, line);
				tokens = tokenize_string(line, ',', PORELEN);
				
				for(j = 0; j < PORELEN; j++) accum_stats_singl[i][j][k] = stold(tokens[j]);
				
			}
			
		}
		
		// Read the accumulated statistics for the single sites
		for(i = 0; i < MTSIZE; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', NATOMS);
			
			for(j = 0; j < NATOMS; j++) gammat[i][j] = stold(tokens[j]);
			
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
	int i, j, k;
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
	
	// Save the time variables
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << time_ell;
	
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << "," <<  nSims << endl;
	
	// Save the atom types of the pore at the moment
	for(i = 0; i < PORELEN; i++){
		
		fileS.precision(std::numeric_limits<int>::max_digits10);
		if(i != 0) fileS << ",";
		fileS << atom_types[i];
		
	}
	fileS << endl;
	
	//Save the accumulated single site statistics
	for(i = 0; i < MTSIZE; i++){
		
		for(k = 0; k < NATOMS; k++){
			
			for(j = 0; j < PORELEN; j++){
				
				fileS.precision(std::numeric_limits<long double>::max_digits10);
				if(j != 0) fileS << ",";
				fileS << accum_stats_singl[i][j][k];
				
			}
			fileS << endl;
			
		}
		
	}
	
	//Save the accumulated single site statistics
	for(i = 0; i < MTSIZE; i++){
		
		for(j = 0; j < NATOMS; j++){
			
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			if(j != 0) fileS << ",";
			fileS << gammat[i][j];
			
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
	
	fr_save(to_string(1) + STATE_FILE, to_string(1) +  GEN_STATE_FILE);
	fr_save(to_string(2) + STATE_FILE, to_string(2) +  GEN_STATE_FILE);
	
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
	if (randMv < rates[0]){  // Move an A particle to an adjacent empty site
		
		uniform_int_distribution <int> move_id(1, n_mvs_A);
		site = hop_mvs_A[move_id(generator) - 1];
		
		do_swp(site, AA, EMPTY);
		
	}
	else if (randMv < rates[1]){  // Move a B particle to an adjacent empty site
		
		uniform_int_distribution <int> move_id(1, n_mvs_B);
		site = hop_mvs_B[move_id(generator) - 1];
		
		do_swp(site, BB, EMPTY);
		
	}
	else if (randMv < rates[2]){  // Swap and A  and B particle on adjacent sites
		
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
	else if (randMv < rates[5]){  // React an A particle to make a B particle
		
		uniform_int_distribution <int> move_id(1, n_mvs_rxn_A);
		site = rxn_mvs_A[move_id(generator) - 1];
		
		do_rxn(site, AA);
		
	}
	else if (randMv < rates[6]){  // React a B particle to make an A particle
		
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
 
 Records the occupancy statistics for a site.
 
 PARAMETERS:
 
	- int atS : If the statistics should be taken (true) or averaged (false).
 
 RETURN:
 
*/
void record_stats(bool atS){
	
	// Auxiliary variables
	int i;
	
	if(atS){
		
		while(timeTrack.size() > 0){
			
			if(time_ell < timeTrack[0]) break;
			
			for(i = 0; i < MTSIZE; i++){
				
				if(timeTrack[0] == maxTime[i]){
					
					record_stats_singl(atS, i);
					timeTrack.erase(timeTrack.begin());
					break;
					
				}
				
			}
			
		}
		
	}
	else{
		
		record_stats_singl(atS, 0);
		
	}
	
}

/*
 
 Records the occupancy statistics for a site.
 
 PARAMETERS:
 
	- int atS : If the statistics should be taken (true) or averaged (false).
	- int tIndex : The time index to save the data to.
 
 RETURN:
 
*/
void record_stats_singl(bool atS, int tIndex){
	
	//Auxiliary variables
	int i, j, k, type1;
	long double ntSims, auxA, auxB;
	
	if(atS){
		
		auxA = 0.0L;
		auxB = 0.0L;
		
		for(i = 0; i < PORELEN; i++){
			
			// Get the atom type for the site
			type1 = atom_types[i];
			
			// Record the statistic
			if(type1 != EMPTY) accum_stats_singl[tIndex][i][type1] += 1.0L;
			
			if(type1 == AA) auxA += 1.0L;
			else if(type1 == BB) auxB += 1.0L;
			
		}
		
		if(!(auxA == 0.0L && auxB == 0.0L)){
		
			gammat[tIndex][AA] += (auxA/(auxA+auxB));
			gammat[tIndex][BB] += (auxB/(auxA+auxB));
		
		}
		
	}
	else{
		
		// Get the long double representation of the total number of simulations
		ntSims = (long double) NSIMS;
				
		// Average the statistics
		for (i = 0; i < MTSIZE; i++){
			
			for (k = 0; k < NATOMS; k++){
				
				gammat[i][k] /= ntSims;
				
				for(j = 0; j < PORELEN; j++) accum_stats_singl[i][j][k] /= ntSims;
				
			}
			
		}
		
	}
	
}

//------------------------------------------------------
//   SETUP SYSTEM RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Sets the statistic arrays to zero.
 
 PARAMETERS:
 
 RETURN:
 
*/
void setup_init_vals(){
	
	// Auxiliary variables
	int i, j, k;
	
	// Initialize the arrays
	for(i = 0; i < MTSIZE; i++){
		
		for(k = 0; k < NATOMS; k++){
			
			gammat[i][k] = 0.0L;
			for(j = 0; j < PORELEN; j++) accum_stats_singl[i][j][k] = 0.0L;
			
		}
		
	}
	
}

/*
 
 Setups the pore for a simulation, it is setup to empty by default.
 
 PARAMETERS:
 
	- bool init : If it is the first time the simulation is running.
 
 RETURN:
 
*/
void setup_pore(bool init){
	
	// Auxiliary variables
	int i;
	
	// Add the atoms to the system
	for (i = 0; i < PORELEN; i++) atom_types[i] = EMPTY;
		
	//Update the lists
	for (i = 0; i < PORELEN; i++){
		
		update_lists_swp(i);
		update_lists_rxn(i);
		
	}
	
}

/*
 
 Setups the system for a simulation.
 
 PARAMETERS:
 
	- bool init : If it is the first time the simulation is running.
 
 RETURN:
 
*/
void setup_system(bool init){
	
	// Auxiliary variables
	int i;
	
	// Setup the starting variables
	if(init){
		
		setup_init_vals();
		perc = 0.0L;
		nsSims = 1;
		
	}
	
	// Time variables set to zero
	time_ell = 0.0L;
	
	// Setup the rates to zero initially
	for(i = 0; i < PORELEN; i++){
		
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
		
	}
	
	// Set the rates to zero
	for(i = 0; i < NMOVES; i++) rates[i] = 0.0L;
	
	// Set the number of moves to zero
	n_mvs_A = 0;
	n_mvs_B = 0;
	n_mvs_swp_AB = 0;
	n_mvs_rxn_A = 0;
	n_mvs_rxn_B = 0;
	
	// Setup the pore
	setup_pore(init);
	
	// Reset the time array
	timeTrack.resize(0);
	for(i = 0; i < MTSIZE; i++) timeTrack.push_back(maxTime[i]);
	
	if(init && LOAD_FILE) fr_load_pr();
	
	// Validate the parameters
	validate_parameters();
	
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
	- int part1 : The first kind of particle of the move.
	- int part2 : The second kind of particle of the move.
	- int mType : The kind of move.
 
 RETURN:
 
*/
void validate_move(int site, int part1, int part2, int mType){
	
	// Auxiliary variables
	int type1, type2;
	
	if(mType == MSWAP){
		
		//Check that the particles are valid to swap
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
		
		//Check that the particles are valid to swap
		if (!((type1 == part1 && type2 == part2) || (type1 == part2 && type2 == part1))){
			
			char a;
			
			cout << "The particles that are being swapped are of the same kind, or not valid: " << endl;
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
		
		//Check that the particles are valid to swap
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
		
		//Check that the particles are valid to swap
		if (type1 != part1){
			
			char a;
			
			cout << "The particle that is reacting are not valid: " << endl;
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
	
	// Number of simulations
	(*fileName) << ",Number of simulations = " << NSIMS;
	
	//End the line
	(*fileName) << endl;
	
}

/*
 
 Writes the different required statistics in the file.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file(){
	
	// Write the specific statistics
	write_info_to_file_singl();
	write_info_to_file_gamma();
	
	// Write the symmetrized statistics
	write_info_to_file_singl_sym();
	
}

/*
 
 Writes the statistics on the average number of atoms of each type in the pore,
 for each time.
 
 PARAMETERS:
 
 RETURN:
 
*/
void write_info_to_file_gamma(){
	
	//Auxiliary variables
	int i;
	ofstream myFile;
	long double t, gammat1,gammat2;
	
	// Open the file and write the file prelude
	myFile.open(FL_NM_GAMMA);
	write_file_prelude(&myFile);
	
	// Write the labels
	myFile << "t" << "," << "NA/NT" << "," << "NB/NT" << endl;   
	
	for(i = 0; i < MTSIZE; i++){
		
		t = maxTime[i];
		gammat1 = gammat[i][AA];
		gammat2 = gammat[i][BB];
		
		myFile << t << "," << gammat1 << "," << gammat2 << endl;
		
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
	int i, j;
	ofstream myFile;
	long double avgs_A, avgs_B, avgs_X, avgs_E;
	
	// Open the file and write the file prelude
	myFile.open(FL_NM_SINGL);
	write_file_prelude(&myFile);
	
	// Write the time prelude
	myFile << "n";   
	
	for(i = 0; i < MTSIZE; i++){
		
		myFile << "," << "<A_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<B_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<X_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<E_n>(t = " << maxTime[i] << ")";
		
	}
	myFile << endl;
	
	for(j = 0; j < PORELEN; j++){
		
		for(i = 0; i < MTSIZE; i++){
			
			if(i == 0) myFile << (j+1);
			
			avgs_A = accum_stats_singl[i][j][AA];
			avgs_B = accum_stats_singl[i][j][BB];
			avgs_X =(avgs_A + avgs_B);
			avgs_E = 1.0L - avgs_X;
			
			myFile << "," << avgs_A << "," << avgs_B << "," << avgs_X << "," << avgs_E;
			
		}
		myFile << endl;
		
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
	int i, j;
	ofstream myFile;
	long double avgs_A, avgs_B, avgs_X, avgs_E;
	
	// Open the file and write the file prelude
	myFile.open(FL_NM_SINGL_SYM);
	write_file_prelude(&myFile);
	
	// Write the time prelude
	myFile << "n";   
	
	for(i = 0; i < MTSIZE; i++){
		
		myFile << "," << "<A_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<B_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<X_n>(t = " << maxTime[i] << ")";
		myFile << "," << "<E_n>(t = " << maxTime[i] << ")";
		
	}
	myFile << endl;
	
	for(j = 0; j < PORELEN; j++){
		
		for(i = 0; i < MTSIZE; i++){
			
			if(i == 0) myFile << (j+1);
			
			avgs_A = 0.5L * (accum_stats_singl[i][j][AA] + accum_stats_singl[i][PORELEN - 1 - j][AA]);
			avgs_B = 0.5L * (accum_stats_singl[i][j][BB] + accum_stats_singl[i][PORELEN - 1 - j][BB]);
			avgs_X =(avgs_A + avgs_B);
			avgs_E = 1.0L - avgs_X;
			
			myFile << "," << avgs_A << "," << avgs_B << "," << avgs_X << "," << avgs_E;
			
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
 
 Returns the percentage of the done simulation.
 
 PARAMETERS:
 
	- int simNum : The simulation number.
 
 RETURN:
 
*/
void sim_percentage(int simNum){
	
	//Auxiliary variables
	int i;
	string stP;
	long double sN;
	
	if(simNum == 0){
		
		cout << "\t";
		for(i = 1; i <= 50; i++) cout << "x";
		for(i = 51; i <= 100; i++) cout << "o";
		cout << endl << "\t";
		
	}
	else{
		
		sN = 100.0L * ((long double) simNum/NSIMS);
		
		while(perc < sN){
			
			perc += 1.0L;
			if(perc <= 50.0L) cout << "x";
			else if(perc > 50.0L) cout << "o";
			cout.flush();
			
			if(SAVE_FILE) fr_save_pr();
			
		}
		
	}
	
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
