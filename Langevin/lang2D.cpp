/*
 
 Title: Langevin simulation in two dimensions.
 
 Author: Andres Garcia.
 
 Description: Langevin simulation to calculate the passing propensity of two
 polyatomic molecules made of circles.
 
 Date: March 2 / 2016.
 
*/

//------------------------------------------------------
//   MODULES AND IMPORTS
//------------------------------------------------------

#include <algorithm>  // Has the functions for Min and Max (see documentation)
#include <array>  // Contains the library for manipulating arrays
#include <cmath>  // Trigonometric, exponential, logarithmic functions, etc.
#include <fstream>  // To write the info in a file
#include <iostream>  // Basic input/output file
#include <random>  // Includes the random number generators (we want the Mersenne Twister, mt19937_64)
#include <sstream>  // Creates a special input/output stream
#include <string>  // To be able to manipulate strings
#include <vector>  // The proper package with the functions to manipulate vectors
#include <time.h>  // Contains time functions, to seed the generator

using namespace std;

//------------------------------------------------------
//   RANDOM NUMBER GENERATOR(S)
//------------------------------------------------------

random_device rd;  // Get a random seed from a generic random device
unsigned int seed = rd();  // Get a random seed
mt19937_64 generator(seed);  // Get the number generator (the 64 bit Mersenne Twister)
normal_distribution<long double> rand_gauss(0.0L, 1.0L);  // Define the Gaussian distribution
uniform_real_distribution<long double> rand_real1(0.0L, 1.0L);  // A uniform real valued distribution

//------------------------------------------------------
//  FILE NAMES AND SAVE OPTIONS
//------------------------------------------------------

// File name to save
const string FL_NAME_BASE = "lang2D";
const string FILE_SAVE_CONF = FL_NAME_BASE + "-mlConfig.csv";
const string FILE_SAVE_NAME = FL_NAME_BASE + "-2Dresults.csv";

//------------------------------------------------------
//   CONSTANTS
//------------------------------------------------------

// Pi = 3.14159265.....
const long double PI = M_PI; 

// Molecule identifiers
const int M1_NUM = 1;
const int M2_NUM = 2;
const int M12_NUM = 3;

// Copy to or from identifiers
const int RESETINIT = 1;
const int COPYTOTEMP = 2;
const int COPYFROMTEMP = 3;

// If the variables should be set to zero
const bool TO_ZERO = true;

// Number of simulations over which to average
const int MAXCNTR = 5000000;

// Number of trials after which to save
const int SAVEAFT = (int) (MAXCNTR / 1000);

// Strings that contain the file names of the molecules to load
const string ST_M1 = "monomer.csv";
const string ST_M2 = "trimer.csv";

// Time and diameter vectors
vector <long double> timesExam = {0.0001L};
vector <long double> diamsExam = {4.4L};

//------------------------------------------------------
//   GENERAL PARAMETERS
//------------------------------------------------------

// Auxiliary counters
int cntT; 
int cntW;

// Pass/fail parameters
int fails;
int passes;

// The save counter
int saveCnt;
bool firstTime;

// If a simulation is to be loaded
bool loadSim;

// Percentage tracker
long double perc;

// The maximum phase space possible for a molecule to move
long double TOLLIM1;
long double TOLLIM2;

// The maximum angle that the molecules can rotate
long double MAXANGL1;
long double MAXANGL2;

// The success or fail condition
long double failPass;
long double successPass;

// The time step to be used
long double TIMESTEP;

//------------------------------------------------------
//   PARAMETERS FOR MOLECULE 1
//------------------------------------------------------

const int N1 = 1;  // Number of atoms for molecule 1

string mol1Name;  // The name of molecule 1

long double boundR1;  // Bounding radius for molecule 1
long double Xcm1[2];  // Array that stores the position of the center of mass of molecule 1
long double X1[N1][2];  // Array that stores the position vectors of the atoms in molecule 1
long double Xor1[2][2];  // Array that stores the orientation vectors of molecule 1
long double radii1[N1];  // Radii of the atoms in molecule 1

// Difusion tensor of molecule 1, D1 = (D_1_x, D_1_y, D_1_theta) 
long double D1[3];  

long double Xcm1i[2];  // Array that stores the position of the center of mass of molecule 1 initially
long double X1i[N1][2];  // Array that stores the position vectors of the atoms in molecule 1 initially
long double Xor1i[2][2];  // Array that stores the orientation vectors of molecule 1 initially

long double Xcm1T[2];  // Array that stores the position of the center of mass of molecule 1 temporarily
long double X1T[N1][2];  // Array that stores the position vectors of the atoms in molecule 1 temporarily
long double Xor1T[2][2];  // Array that stores the orientation vectors of molecule 1 temporarily

//------------------------------------------------------
//   PARAMETERS FOR MOLECULE 2
//------------------------------------------------------

const int N2 = 3;  // Number of atoms for molecule 2

string mol2Name;  // The name of molecule 2

long double boundR2;  // Bounding radius for molecule 2
long double Xcm2[2];  // Array that stores the position of the center of mass of molecule 2
long double X2[N2][2];  // Array that stores the position vectors of the atoms in molecule 2
long double Xor2[2][2];  // Array that stores the orientation vectors of molecule 2
long double radii2[N2];  // Radii of the atoms in molecule 2

// Difusion tensor of molecule 2, D2 = (D_2_x, D_2_y, D_2_theta)
long double D2[3];  

long double Xcm2i[2];  // Array that stores the position of the center of mass of molecule 2 initially
long double X2i[N2][2];  // Array that stores the position vectors of the atoms in molecule 2 initially
long double Xor2i[2][2];  // Array that stores the orientation vectors of molecule 2 initially

long double Xcm2T[2];  // Array that stores the position of the center of mass of molecule 2 temporarily
long double X2T[N2][2];  // Array that stores the position vectors of the atoms in molecule 2 temporarily
long double Xor2T[2][2];  // Array that stores the orientation vectors of molecule 2 temporarily

//------------------------------------------------------
//   FILE OPERATION CONSTANTS
//------------------------------------------------------

//Constants for the different operations
const int OPEN_FILE = 0;
const int WRITE_FILE = 1;
const int NEW_LINE_FILE = 2;
const int OPEN_FILE_OVER = 3;

//------------------------------------------------------
//   PORE PARAMETERS
//------------------------------------------------------

long double poreTop;  // Denotes the coordinate at the top of the pore
long double poreBot;  // Denotes the coordinate at the bottom of the pore
long double poreWidth;  // Denotes the width of the pore

//------------------------------------------------------
//   RESULTS PARAMETERS
//------------------------------------------------------

// Get the size for the array of results
vector <vector <long double> > timesAndWidths;  // The array that keeps track of the examined times

//------------------------------------------------------
//   SAVE AND LOAD FILE CONSTANTS AND VARIABLES
//------------------------------------------------------

//Name of the file where to save the simulation state
const string STATE_FILE = FL_NAME_BASE + "-state.csv";
const string GEN_STATE_FILE = FL_NAME_BASE + "-gState.txt";

// If the simulation should be saved/loaded
const bool SAVE_FILE = true;
const bool LOAD_FILE = false;

//------------------------------------------------------
//   FUNCTION PROTOTYPES
//------------------------------------------------------

// Copy array related subroutines
void copy_positions(int, int);

// File save and load related subroutines
void fr_load(string, string);
void fr_load_pr();
void fr_save(string, string);
void fr_save_pr();

// Get information related subroutines
long double get_angle_max(int);
long double get_disp(long double);
long double get_max_y_pos(long double [][2], long double [], int);
long double get_min_y_pos(long double [][2], long double [], int);
long double get_position_prob(int);
void get_prob(long double);
long double get_tollim(int);

// Rotate molecules related subroutines
void rotate_mol(int);
void rotate_vec(long double *, long double);

// Setup related subroutines
void setup_initial(int *);
void setup_mols(bool);
void setup_mols_initi();
void setup_mols_trial();
void setup_trial(long double, bool);

// Translate molecules related subroutines
void translate_mol(int);
void translate_vec(long double *, long double, long double);

// Validation related subroutines
bool validate_intersect(int);

// Write files related subroutines
void write_file_prelude(ofstream *);
void write_mol_config(int);
void write_prob_results(int, long double);

// Other subroutines
int int_pow(int, int);
long double int_pow_L(long double, int);
long double max_of_array(long double, int);
long double min_of_array(long double, int);
void print_mol_info(int);
void sim_percentage(int);
vector <string> tokenize_string(string, char, int);

//------------------------------------------------------
//   MAIN FUNCTION
//------------------------------------------------------

/*
 
 Main program, runs the code until it's done.
 
*/
int main(){
	
	// Auxiliary variables
	int initSize1, initSize2;
	
	cout << "Langevin Circle 2D Simulation: " << FL_NAME_BASE << endl;
	cout << "Program starts!" << endl;
	
	// Setup the initial simulation
	cntT = 0;
	setup_initial(&initSize1);
	
	//-------------------------------------   
	// Run the algorithm
	//-------------------------------------
	
	while(timesAndWidths.size() > 0){
		
		// Set the counters
		cntT++;
		cntW = 0;
		initSize2 = ((int) timesAndWidths[0].size()) - 1;
		
		// Set the time step
		TIMESTEP = timesAndWidths[0][0];
		
		while(timesAndWidths[0].size() > 1){
			
			// Currently running
			cntW++;
			cout << "Time step " << cntT << " of " << initSize1 << " and pore radius " << cntW << " of " << initSize2 << endl;
			cout << "Running simulation with time step: " << TIMESTEP << ", and pore radius: "<< (0.5L * timesAndWidths[0][1]) << endl;
			cout.flush();
			
			// Get the probability
			get_prob(timesAndWidths[0][1]);
			
			// Write the results for the probability
			timesAndWidths[0].erase(timesAndWidths[0].begin() + 1);
			
			// Set a new line for the next set of results
			write_prob_results(NEW_LINE_FILE, 0.0L);
			
		}
		
		// Write the probability for the specific set of results and reset the array
		timesAndWidths.erase(timesAndWidths.begin());
		
	}
	
	cout << "Done running the program!" << endl;
	
	return 0;
	
}

//------------------------------------------------------
//  COPY ARRAYS RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Copies the positions of the position vectors to the temporary vectors or vice-versa.
 
 PARAMETERS:
 
	- int oper : The operation to be performed; copy from the temporary arrays (COPYFROMTEMP), copy to the temporary arrays (COPYTOTEMP) .
	- int molNum : The molecule on which the operation should be made; first molecule (M1_NUM), second molecule (M2_NUM).
 
 RETURN:
 
*/
void copy_positions(int oper, int molNum){
	
	// Auxiliary variables
	int i, j;
	
	if(oper == COPYTOTEMP){
		
		// Copy from the first molecule
		if(molNum == M1_NUM || molNum == M12_NUM){
			
			for(i = 0; i < 2; i++){
				
				Xcm1T[i] = Xcm1[i];
				
				for(j = 0; j < N1; j++) X1T[j][i] = X1[j][i];
				for(j = 0; j < 2; j++) Xor1T[i][j] = Xor1[i][j];
				
			}
			
		}
		
		// Copy from the second molecule
		if(molNum == M2_NUM || molNum == M12_NUM){
			
			for(i = 0; i < 2; i++){
				
				Xcm2T[i] = Xcm2[i];
				
				for(j = 0; j < N2; j++) X2T[j][i] = X2[j][i];
				for(j = 0; j < 2; j++) Xor2T[i][j] = Xor2[i][j];
				
			}
			
		}
		
	}
	else if(oper == COPYFROMTEMP){
		
		// Copy to the first molecule
		if(molNum == M1_NUM || molNum == M12_NUM){
			
			for(i = 0; i < 2; i++){
				
				Xcm1[i] = Xcm1T[i];
				
				for(j = 0; j < N1; j++) X1[j][i] = X1T[j][i];
				for(j = 0; j < 2; j++) Xor1[i][j] = Xor1T[i][j];
				
			}
			
		}
		
		// Copy to the second molecule
		if(molNum == M2_NUM || molNum == M12_NUM){
			
			for(i = 0; i < 2; i++){
				
				Xcm2[i] = Xcm2T[i];
				
				for(j = 0; j < N2; j++) X2[j][i] = X2T[j][i];
				for(j = 0; j < 2; j++) Xor2[i][j] = Xor2T[i][j];
				
			}
			
		}
		
	}
	else if(oper == RESETINIT){
		
		// Return molecule 1 to the original configuration
		for(i = 0; i < 2; i++){
			
			Xcm1[i] = Xcm1i[i];
			
			for(j = 0; j < N1; j++) X1[j][i] = X1i[j][i];
			for(j = 0; j < 2; j++) Xor1[i][j] = Xor1i[i][j];
			
		}
		
		// Return molecule 2 to the original configuration
		for(i = 0; i < 2; i++){
			
			Xcm2[i] = Xcm2i[i];
			
			for(j = 0; j < N2; j++) X2[j][i] = X2i[j][i];
			for(j = 0; j < 2; j++) Xor2[i][j] = Xor2i[i][j];
			
		}
		
	}
	
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
	vector <long double> aV;
	
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
		
		//-----------------------------------------
		// LOAD THE BASIC DATA
		//-----------------------------------------
		
		// Read the save counter
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		saveCnt = stoi(tokens[0]);
		
		// Read the pass and fail counters
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		passes = stoi(tokens[0]);
		fails = stoi(tokens[1]);
		
		// Validate the loaded number
		if((passes + fails) > MAXCNTR){
			
			cout << "The amount of trial exceeds the maximum number of trials, results will be wrong:" << endl;
			cout << "\tNumber of trials: " << (passes + fails) << endl;
			cout << "\tMaximum Number of trials: " <<  MAXCNTR << endl;
			
			char a;
			cout << "The program will terminate, enter any character to continue: ";
			cin >> a;
			exit(EXIT_FAILURE);
			
		}
		
		// Read the failPass and successPass conditions
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		failPass = stold(tokens[0]);
		successPass = stold(tokens[1]);
		
		// Read the amount of times and widths to be examined, reset the array
		getline(fileS, line);
		timesAndWidths.resize(0);
		tokens = tokenize_string(line, ',', 1);
		
		j = stoi(tokens[0]);
		
		// Read the times and widths to be examined
		for(i = 0; i < j; i++){
			
			aV.resize(0);
			getline(fileS, line);
			tokens = tokenize_string(line, ',', -1);
			
			for(k = 0; k < (int) tokens.size(); k++) aV.push_back(stold(tokens[k]));
			
			// Add it to the time and Widths array
			timesAndWidths.push_back(aV);
			
		}
		
		//-----------------------------------------
		// LOAD THE INFORMATION FOR MOLECULE 1
		//-----------------------------------------
		
		// Read the name of the molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		mol1Name = tokens[0];
		
		// Read the number of atoms in the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		// Validate the number of atoms
		if(N1 != stoi(tokens[0])){
			
			cout << "The amount of atoms in molecule 1 don't match with the loaded number of atoms:" << endl;
			cout << "Number of atoms N1: " << N1 << endl;
			cout << "Number of atoms to be loaded: " <<  stoi(tokens[0]) << endl;
			
			char a;
			cout << "The program will terminate, enter any character to continue: ";
			cin >> a;
			exit(EXIT_FAILURE);
			
		}
		
		//+++++++++++++++
		// Initial data
		//+++++++++++++++
		
		// Read the initial center of mass of the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		for(i = 0; i < 2; i++) Xcm1i[i] = stold(tokens[i]);
		
		// Read the initial position of the atoms in the first molecule
		for(i = 0; i < N1; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) X1i[i][j] = stold(tokens[j]);
			
		}
		
		// Read the initial orientation of the atoms in the first molecule
		for(i = 0; i < 2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) Xor1i[i][j] = stold(tokens[j]);
			
		}
		
		//+++++++++++++++
		// In sim data
		//+++++++++++++++
		
		// Read the initial center of mass of the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		for(i = 0; i < 2; i++) Xcm1[i] = stold(tokens[i]);
		
		// Read the initial position of the atoms in the first molecule
		for(i = 0; i < N1; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) X1[i][j] = stold(tokens[j]);
			
		}
		
		// Read the initial orientation of the atoms in the first molecule
		for(i = 0; i < 2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) Xor1[i][j] = stold(tokens[j]);
			
		}
		
		//+++++++++++++++++++++++
		// Other relevant data
		//+++++++++++++++++++++++
		
		// Read the bounding radius of the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		boundR1 = stold(tokens[0]);
		
		// Read the radii of the atoms in the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', N1);
		
		for(i = 0; i < N1; i++) radii1[i] = stold(tokens[0]);
		
		// Read the diffusion tensor of the first molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 3);
		
		for(i = 0; i < 3; i++) D1[i] = stold(tokens[i]);
		
		//-----------------------------------------
		// LOAD THE INFORMATION FOR MOLECULE 2
		//-----------------------------------------
		
		// Read the name of the molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		mol2Name = tokens[0];
		
		// Read the number of atoms in the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		// Validate the number of atoms
		if(N2 != stoi(tokens[0])){
			
			cout << "The amount of atoms in molecule 2 don't match with the loaded number of atoms:" << endl;
			cout << "Number of atoms N2: " << N2 << endl;
			cout << "Number of atoms to be loaded: " <<  stoi(tokens[0]) << endl;
			
			char a;
			cout << "The program will terminate, enter any character to continue: ";
			cin >> a;
			exit(EXIT_FAILURE);
			
		}
		
		//+++++++++++++++
		// Initial data
		//+++++++++++++++
		
		// Read the initial center of mass of the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		for(i = 0; i < 2; i++) Xcm2i[i] = stold(tokens[i]);
		
		// Read the initial position of the atoms in the second molecule
		for(i = 0; i < N2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) X2i[i][j] = stold(tokens[j]);
			
		}
		
		// Read the initial orientation of the atoms in the second molecule
		for(i = 0; i < 2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) Xor2i[i][j] = stold(tokens[j]);
			
		}
		
		//+++++++++++++++
		// In sim data
		//+++++++++++++++
		
		// Read the initial center of mass of the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 2);
		
		for(i = 0; i < 2; i++) Xcm2[i] = stold(tokens[i]);
		
		// Read the initial position of the atoms in the second molecule
		for(i = 0; i < N2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) X2[i][j] = stold(tokens[j]);
			
		}
		
		// Read the initial orientation of the atoms in the second molecule
		for(i = 0; i < 2; i++){
			
			getline(fileS, line);
			tokens = tokenize_string(line, ',', 2);
			
			for(j = 0; j < 2; j++) Xor2[i][j] = stold(tokens[j]);
			
		}
		
		//+++++++++++++++++++++++
		// Other relevant data
		//+++++++++++++++++++++++
		
		// Read the bounding radius of the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 1);
		
		boundR2 = stold(tokens[0]);
		
		// Read the radii of the atoms in the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', N2);
		
		for(i = 0; i < N2; i++) radii2[i] = stold(tokens[0]);
		
		// Read the diffusion tensor of the second molecule
		getline(fileS, line);
		tokens = tokenize_string(line, ',', 3);
		
		for(i = 0; i < 3; i++) D2[i] = stold(tokens[i]);
		
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
	
	// Auxiliary variables
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
	
	//-----------------------------------------
	// SAVE GENERATOR AND THE SEED USED
	//-----------------------------------------
	
	//Open the generator state file and overwrite it, if it's the case
	fileS.open(geString);
	
	// Save the generator and the seed
	fileS << generator << endl;
	fileS << seed << endl;
	
	// Close the file
	fileS.close();
	
	//-----------------------------------------
	// SAVE THE BASIC DATA
	//-----------------------------------------
	
	// Save the basic data to a file
	fileS.open(stString);
	
	// The save counter
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << saveCnt << endl;
	
	// The number of passes and failures
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << passes << "," << fails << endl;
	
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << failPass << "," << successPass << endl;
	
	// Save the time and widths array, first save the size
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << timesAndWidths.size() << endl;
	
	for(i = 0 ; i < (int) timesAndWidths.size(); i++){
		
		for(j = 0; j < (int) timesAndWidths[i].size(); j++){ 
			
			if(j != 0) fileS << ",";
			fileS.precision(std::numeric_limits<long double>::max_digits10);
			fileS << timesAndWidths[i][j];
			
		}
		fileS << endl;
		
	}
	
	//-----------------------------------------
	// SAVE THE INFORMATION FOR MOLECULE 1
	//-----------------------------------------
	
	// Save the name of the molecule
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << mol1Name << endl;
	
	// Save the number of atoms in molecule 1
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << N1 << endl;
	
	//+++++++++++++++
	// Initial data
	//+++++++++++++++
	
	// Save the initial position for the center of mass, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << Xcm1i[0] << "," << Xcm1i[1] << endl;
	
	// Save the initial position of the atoms, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < N1; i++) fileS << X1i[i][0] << "," << X1i[i][1] << endl;
	
	// Save the initial orientation of the movement axes of the atoms, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < 2; i++) fileS << Xor1i[i][0] << "," << Xor1i[i][1] << endl;
	
	//+++++++++++++++
	// In sim data
	//+++++++++++++++
	
	// Save the current position for the center of mass, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << Xcm1[0] << "," << Xcm1[1] << endl;
	
	// Save the current position of the atoms, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < N1; i++) fileS << X1[i][0] << "," << X1[i][1] << endl;
	
	// Save the current orientation of the movement axes of the atoms, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < 2; i++) fileS << Xor1[i][0] << "," << Xor1[i][1] << endl;
	
	//+++++++++++++++++++++++
	// Other relevant data
	//+++++++++++++++++++++++
	
	// Save the bounding radius, for molecule 1
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << boundR1 << endl;
	
	// Save the radii of the atoms, for molecule 1
	for(i = 0; i < N1; i++){
		
		if(i != 0) fileS << ",";
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		fileS << radii1[i];
		
	}
	fileS << endl;
	
	// Save the diffusion tensor, for molecule 1
	for(i = 0; i < 3; i++){
		
		if(i != 0) fileS << ",";
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		fileS << D1[i];
		
	}
	fileS << endl;
	
	//-----------------------------------------
	// SAVE THE INFORMATION FOR MOLECULE 2
	//-----------------------------------------
	
	// Save the name of the molecule
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << mol2Name << endl;
	
	// Save the number of atoms in molecule 2
	fileS.precision(std::numeric_limits<int>::max_digits10);
	fileS << N2 << endl;
	
	//+++++++++++++++
	// Initial data
	//+++++++++++++++
	
	// Save the initial position for the center of mass, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << Xcm2i[0] << "," << Xcm2i[1] << endl;
	
	// Save the initial position of the atoms, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < N2; i++) fileS << X2i[i][0] << "," << X2i[i][1] << endl;
	
	// Save the initial orientation of the movement axes of the atoms, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < 2; i++) fileS << Xor2i[i][0] << "," << Xor2i[i][1] << endl;
	
	//+++++++++++++++
	// In sim data
	//+++++++++++++++
	
	// Save the current position for the center of mass, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << Xcm2[0] << "," << Xcm2[1] << endl;
	
	// Save the current position of the atoms, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < N2; i++) fileS << X2[i][0] << "," << X2[i][1] << endl;
	
	// Save the current orientation of the movement axes of the atoms, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	for(i = 0; i < 2; i++) fileS << Xor2[i][0] << "," << Xor2[i][1] << endl;
	
	//+++++++++++++++++++++++
	// Other relevant data
	//+++++++++++++++++++++++
	
	// Save the bounding radius, for molecule 2
	fileS.precision(std::numeric_limits<long double>::max_digits10);
	fileS << boundR2 << endl;
	
	// Save the radii of the atoms, for molecule 2
	for(i = 0; i < N2; i++){
		
		if(i != 0) fileS << ",";
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		fileS << radii2[i];
		
	}
	fileS << endl;
	
	// Save the diffusion tensor, for molecule 2
	for(i = 0; i < 3; i++){
		
		if(i != 0) fileS << ",";
		fileS.precision(std::numeric_limits<long double>::max_digits10);
		fileS << D2[i];
		
	}
	fileS << endl;
	
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
//   GET INFORMATION RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Gets the maximum angle of rotation for a molecule.
 
 PARAMETERSs:
 
	- int molNum : The id of the molecule for which to get the angle.
 
 RETURN:
 
	- long double maxAngl : The maximum angle of rotation for the molecule.
 
*/
long double get_angle_max(int molNum){
	
	//Auxiliary variables
	long double maxAngle, maxTol;
	
	// Initialize the variables
	maxAngle = PI/2.0L;
	
	if(molNum == M1_NUM){
		
		// If the pore is wide enough, then it can rotate 2Pi degrees
		maxTol = 2.0L * (long double) N1;
		
		if(N1 > 1 && poreWidth < maxTol) maxAngle = asin((poreWidth - 2.0)/(maxTol- 2.0L));
		
	}
	else if(molNum == M2_NUM){
		
		// If the pore is wide enough, then it can rotate 2Pi degrees
		maxTol = 2.0L * (long double) N2;
		
		if(N2 > 1 && poreWidth < maxTol) maxAngle = asin((poreWidth - 2.0)/(maxTol - 2.0L));
		
	}
	
	return maxAngle;
	
}

/*
 
 Gets the displacement based on the diffusion tensor in a specific direction.
 
 PARAMETERSs:
 
	- long double diffD : The diffusion coefficient in the direction of motion.
 
 RETURN:
 
	- long double disp : The displacement generated by the given parameters.
 
*/
long double get_disp(long double diffD){
	
	//Auxiliary variables
	long double disp;
	
	disp = rand_gauss(generator) * sqrt(2.0L * diffD * TIMESTEP);
	
	return disp;
	
}

/*
 
 Gets the difference in position of the top pore wall to the top most atom in a molecule.
 
 PARAMETERS:
 
	- long double [][2] M1 : A matrix with N1 rows of 2 components. The ith row represent the 2D position of the ith atom of a molecule.
	- long double [] rad1 : The raddii of the the atoms of the molecule.
	- int size1 : The number of atoms in M1.
 
 RETURN:
 
	- long double disp : The difference in position of the wall to the top atom.
 
*/
long double get_max_y_pos(long double M1[][2], long double rad1[], int size1){
	
	//Auxiliary variables
	int i;
	long double disp;
	
	// Set the y-maximum position to the one of the first atom
	disp = M1[0][1] + rad1[0];
	
	// Get the top coordinate
	for(i = 1; i < size1; i++){
		
		if(disp < (M1[i][1]+rad1[i])) disp = M1[i][1] + rad1[i];
		
	}
	
	disp = (poreTop - disp);
	
	return disp;
	
}

/*
 
 Gets the difference in position of the bottom pore wall to the bottom most atom in a molecule.
 
 PARAMETERS:
 
	- long double [][2] M1 : A matrix with N1 rows of 2 components. The ith row represent the 2D position of the ith atom of a molecule.
	- long double [] rad1 : The raddii of the the atoms of the molecule.
	- int size1 : The number of atoms in M1.
 
 RETURN:
 
	- long double disp : The difference in position of the wall to the top atom.
 
*/
long double get_min_y_pos(long double M1[][2], long double rad1[], int size1){
	
	//Auxiliary variables
	int i;
	long double disp;
	
	// Set the y-minimum position to the one of the first atom
	disp = M1[0][1] - rad1[0];
	
	// Get the top coordinate
	for(i = 1; i < size1; i++){
		
		if(disp > (M1[i][1]-rad1[i])) disp = M1[i][1] - rad1[i];
		
	}
	
	// Get the displacement
	disp = (poreBot - disp);
	
	return disp;
	
}

/*
 
 Gets the probability of the molecule being in a specific orientation.
 
 PARAMETERS:
 
	- int molNum : The id of the molecule.
 
 RETURN:
 
	- long double pProb : The probability with which the molecule can be placed in that position.
 
*/
long double get_position_prob(int molNum){
	
	// Auxiliary variables
	long double mB, mT, pProb;
	
	// Initialize the variable
	pProb = 1.0L;
	
	if(molNum == M1_NUM){
		
		if(N1 > 1){
			
			// Get the top and bottom position
			mT = get_max_y_pos(X1, radii1, N1);
			mB = get_min_y_pos(X1, radii1, N1);
			
			// Get the available phase space for the molecule to move
			mT = (mT - mB);
			
			// Get the orientation probability
			pProb = mT/TOLLIM1;
			
		}
		
	}
	else if(molNum == M2_NUM){
		
		if(N2 > 1){
			
			// Get the top and bottom position
			mT = get_max_y_pos(X2, radii2, N2);
			mB = get_min_y_pos(X2, radii2, N2);
			
			// Get the available phase space for the molecule to move
			mT = (mT - mB);
			
			// Get the orientation probability
			pProb = mT/TOLLIM2;
			
		}
		
	}
	
	// Return the probability
	return pProb;
	
}

/*
 
 Gets the passing probability for a specific pore width.
 
 PARAMETERS:
 
	- long double poreW : The width of the pore to simulate.
 
 RETURN:
 
*/
void get_prob(long double poreW){
	
	// Auxiliary variables
	bool cond1, cond2;
	
	// Setup the system initially
	setup_trial(poreW, TO_ZERO);
	
	write_mol_config(OPEN_FILE);
	write_mol_config(WRITE_FILE);
	
	while((passes + fails) < MAXCNTR){
		
		// Attempt to translate and rotate the molecules
		do{
			
			//Copy the arrays to the auxiliary arrays
			copy_positions(COPYTOTEMP, M12_NUM);
			
			// Translate and rotate molecule 1
			translate_mol(M1_NUM);
			if(N1 > 1) rotate_mol(M1_NUM);
			
			// Translate and rotate molecule 2
			translate_mol(M2_NUM);
			if(N2 > 1) rotate_mol(M2_NUM);      
			
			// Evaluate the continuing conditions
			cond1 = validate_intersect(M12_NUM);
			cond1 = (cond1 || validate_intersect(M1_NUM));
			cond1 = (cond1 || validate_intersect(M2_NUM));
			
			// If the molecules don't intersect, continue with the simulation
			if(!cond1) break;
			
			// Get the positions back from the temporary vector
			copy_positions(COPYFROMTEMP, M12_NUM);
			
		}while(true);
		
		write_mol_config(WRITE_FILE);
		
		// Get the passing conditions
		cond1 = ((Xcm1[0]-Xcm2[0]) <= successPass);
		cond2 = ((Xcm1[0]-Xcm2[0]) >= failPass);
		
		// See if the simulation should be continued or reset
		if(cond1) passes += 1;
		else if(cond2) fails += 1;
		
		if(cond1 || cond2) setup_trial(poreW, !TO_ZERO);
		
		sim_percentage(passes + fails);
		
	}
	
}

/*
 
 Gets the passing probability for a specific pore width.
 
 PARAMETERS:
 
	- int molNum : The molecule for which the tolerance limit should be obtained.
 
 RETURN:
 
	- long double tolLim : The tolerance limit of the molecule.
 
*/
long double get_tollim(int molNum){
	
	// Auxiliary variables
	int i;
	long double dotP, maxT, maxTA, minT, minTA, theta, tolLim;
	
	// Setup the initial values for the variable
	tolLim = -1.0;
	
	if(molNum == M1_NUM){
	
		// Bring the molecule to the center of mass coordinate
		for(i = 0; i < N1; i++){
		
			X1[i][0] -= Xcm1[0];
			X1[i][1] -= Xcm1[1];
		
		}
		
		// Check all the angles
		for(theta = 0.0L; theta <= PI; theta += 0.0001L){
			
			// Project the coordinates on the unit vector
			dotP = cos(theta) * X1[0][0] + sin(theta) * X1[0][1];
			maxT = dotP + radii1[0];
			minT = dotP - radii1[0];
			
			// For all the particles
			for(i = 1; i < N1; i++){
				
				dotP = cos(theta) * X1[i][0] + sin(theta) * X1[i][1];
				maxTA = dotP + radii1[i];
				minTA = dotP - radii1[i];
				
				if(maxT < maxTA) maxT = maxTA;
				if(minT > minTA) minT = minTA;
				
			}
			
			// Get the shortest distance
			if(tolLim < 0.0L) tolLim = maxT - minT;
			else if(tolLim > (maxT - minT)) tolLim = maxT - minT;
									
		}
		
		// Project the coordinates on the unit vector
		dotP = cos(0.5L * PI) * X1[0][0] + sin(0.5L * PI) * X1[0][1];
		maxT = dotP + radii1[0];
		minT = dotP - radii1[0];
		
		// For all the particles
		for(i = 1; i < N1; i++){
			
			dotP = cos(theta) * X1[i][0] + sin(theta) * X1[i][1];
			maxTA = dotP + radii1[i];
			minTA = dotP - radii1[i];
			
			if(maxT < maxTA) maxT = maxTA;
			if(minT > minTA) minT = minTA;
			
		}
		
		// Get the shortest distance
		if(tolLim < 0.0L) tolLim = maxT - minT;
		else if(tolLim > (maxT - minT)) tolLim = maxT - minT;
						
		// Take the molecule back to its original position
		for(i = 0; i < N1; i++){
			
			X1[i][0] += Xcm1[0];
			X1[i][1] += Xcm1[1];
			
		}
	
	}
	else if(molNum == M2_NUM){
		
		// Bring the molecule to the center of mass coordinate
		for(i = 0; i < N2; i++){
			
			X2[i][0] -= Xcm2[0];
			X2[i][1] -= Xcm2[1];
			
		}
		
		// Check all the angles
		for(theta = 0.0L; theta <= PI; theta += 0.001L){
			
			// Project the coordinates on the unit vector
			dotP = cos(theta) * X2[0][0] + sin(theta) * X2[0][1];
			maxT = dotP + radii2[0];
			minT = dotP - radii2[0];
			
			// For all the particles
			for(i = 1; i < N2; i++){
				
				dotP = cos(theta) * X2[i][0] + sin(theta) * X2[i][1];
				maxTA = dotP + radii2[i];
				minTA = dotP - radii2[i];
				
				if(maxT < maxTA) maxT = maxTA;
				if(minT > minTA) minT = minTA;
				
			}
			
			// Get the shortest distance
			if(tolLim < 0.0L) tolLim = maxT - minT;
			else if(tolLim > (maxT - minT)) tolLim = maxT - minT;
			
		}
		
		// Project the coordinates on the unit vector
		dotP = cos(0.5L * PI) * X2[0][0] + sin(0.5L * PI) * X2[0][1];
		maxT = dotP + radii2[0];
		minT = dotP - radii2[0];
		
		// For all the particles
		for(i = 1; i < N2; i++){
			
			dotP = cos(theta) * X2[i][0] + sin(theta) * X2[i][1];
			maxTA = dotP + radii2[i];
			minTA = dotP - radii2[i];
			
			if(maxT < maxTA) maxT = maxTA;
			if(minT > minTA) minT = minTA;
			
		}
		
		// Get the shortest distance
		if(tolLim < 0.0L) tolLim = maxT - minT;
		else if(tolLim > (maxT - minT)) tolLim = maxT - minT;
		
		// Take the molecule back to its original position
		for(i = 0; i < N2; i++){
			
			X2[i][0] += Xcm2[0];
			X2[i][1] += Xcm2[1];
			
		}
				
	}
	
	// Substract the tolerance limit
	tolLim = poreWidth - tolLim;
	
	// Return the value
	return tolLim;
	
}

//------------------------------------------------------
//   ROTATE MOLECULES RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Rotates the given molecule about its center of mass.
 
 PARAMETERS:
 
	- int molNum : The id of the molecule to rotate.
 
 RETURN:
 
*/
void rotate_mol(int molNum){
	
	//Auxiliary variables
	int i;
	long double angl;
	
	// Rotate molecule 1
	if(molNum == M1_NUM || molNum == M12_NUM){
		
		if(N1 <= 1) return;
		
		// Choose the rotation angle   
		angl = get_disp(D1[2]);
		
		// Rotate the molecule
		for(i = 0; i < N1; i++){
			
			// Translate vectors to the center of mass
			translate_vec(&X1[i][0], -Xcm1[0], -Xcm1[1]);
			
			// Rotate the vectors
			rotate_vec(&X1[i][0], angl);
			
			// Translate vectors to their final position
			translate_vec(&X1[i][0], Xcm1[0], Xcm1[1]);
			
		}
		
		// Rotate the axes
		rotate_vec(&Xor1[0][0], angl);
		rotate_vec(&Xor1[1][0], angl);
		
	}
	
	// Rotate molecule 2
	if(molNum == M2_NUM || molNum == M12_NUM){
		
		if(N2 <= 1) return;
		
		// Choose the rotation angle   
		angl = get_disp(D2[2]);
		
		// Rotate the molecule
		for(i = 0; i < N2; i++){
			
			// Translate vectors to the center of mass
			translate_vec(&X2[i][0], -Xcm2[0], -Xcm2[1]);
			
			// Rotate the vectors
			rotate_vec(&X2[i][0], angl);
			
			// Translate vectors to their final position
			translate_vec(&X2[i][0], Xcm2[0], Xcm2[1]);
			
		}
		
		// Rotate the axes
		rotate_vec(&Xor2[0][0], angl);
		rotate_vec(&Xor2[1][0], angl);
		
	}
	
}

/*
 
 Rotates a vector about a point.
 
 PARAMETERS:
 
	- long double * vec : Pointer to the first element of a 2 dimensional vector.
	- long double angle : The angle to rotate the vector about the point.
 
 RETURN:
 
*/
void rotate_vec(long double * vec, long double angle){
	
	// Auxiliary variables
	long double vecAux[2];
	
	// Make a copy of the vector
	vecAux[0] = vec[0];
	vecAux[1] = vec[1];
	
	// Rotate the vector
	vec[0] = cos(angle) * vecAux[0] - sin(angle) * vecAux[1];
	vec[1] = sin(angle) * vecAux[0] + cos(angle) * vecAux[1];
	
}

//------------------------------------------------------
//   SETUP RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Sets up the system for the initial simulation. This includes the pore widths to be examined.
 
 PARAMETERS:
 
	- int * timeSize : Variable where the total number of times is stored.
 
 RETURN:
 
*/
void setup_initial(int * timesSize){
	
	// Auxiliary variables
	int i, j;
	vector <long double> aVt;
	
	// If a simulation is to be loaded
	loadSim = LOAD_FILE;
	
	// Set the save counter to the initial value
	firstTime = true;
	saveCnt = SAVEAFT;
	
	// Empty the times and widths array
	timesAndWidths.resize(0);
	
	for(i = 0; i < (int) timesExam.size(); i++){
		
		aVt.resize(0);
		aVt.push_back(timesExam[i]);
		
		for(j = 0; j < (int) diamsExam.size(); j++) aVt.push_back(diamsExam[j]); 
		timesAndWidths.push_back(aVt);
		
	}
	
	// Load the file or setup the molecules
	if(loadSim) fr_load_pr(); 
	else setup_mols(TO_ZERO);  
	
	// Open the file to write the results with the file prelude
	write_prob_results(OPEN_FILE, 0.0L);
	
	// Set the size of the total number of times to be examined
	(*timesSize) = (int) timesAndWidths.size();
	
}

/*
 
 Sets up molecules for the first time or for a trial.
 
 PARAMETERS:
 
	- bool toZero : If it's the first time the molecules are being setup.
 
 RETURN:
 
*/
void setup_mols(bool toZero){
	
	if(toZero) setup_mols_initi();
	else setup_mols_trial();
	
}

/*
 
 Sets up molecules for the first time and for a trial.
 
 PARAMETERS:
 
 RETURN:
 
*/
void setup_mols_initi(){
	
	// Auxiliary variales
	int i, j;
	string line;
	ifstream myFile;
	vector <string> tokens;
	vector <long double> aV;
	
	try{
		
		//---------------------------------
		// Molecule 1
		//---------------------------------
		
		// Open the file
		myFile.open(ST_M1);
		
		// Read the molecule name
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		mol1Name = tokens[1];
		
		// Read the number of atoms in the molecule
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		if(N1 != stoi(tokens[1])){
			
			//Throw exception if the number of atoms don't match
			cout << "The number of atoms N1 = " << N1 << " must match the number of atoms in the molecule Nm1 = " << stoi(tokens[1]) << endl;
			throw 1;
			
		}
		
		// Set the reading precision to the maximum
		myFile.precision(std::numeric_limits<long double>::max_digits10);
		
		// Read the radius of the bounding sphere
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		
		boundR1 = stold(tokens[1]);
		if(boundR1 < 0.0L) boundR1 = -boundR1;
		
		// Read center of mass
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 3);
		
		for(i = 1; i <= 2; i++) Xcm1i[i-1] = stold(tokens[i]);
		
		// Read the orientation axis
		for(i = 0; i < 2; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 3);
			
			for(j = 1; j <= 2; j++) Xor1i[i][j-1] = stold(tokens[j]);
			
		}
		
		// Read the positions of the atoms in the molecule and their radii
		for(i = 0; i < N1; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 4);
			
			for(j = 1; j <= 2; j++) X1i[i][j-1] = stold(tokens[j]);
			radii1[i] = stold(tokens[3]);
			
		}
		
		// Get the blank line
		getline(myFile, line);
		
		// Read the diffusion tensor
		for(i = 0; i < 3; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 3);
			
			D1[i] = stold(tokens[i]);
			
		}
		
		// Close the file
		myFile.close();
		
	}
	catch(...){
		
		cout << "Failed to read the file for molecule 1. Check the file and try again." << endl;
		exit(EXIT_FAILURE);
		
	}
	
	try{
		
		//---------------------------------
		// Molecule 2
		//---------------------------------
		
		// Open the file
		myFile.open(ST_M2);
		
		// Read the molecule name
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		mol2Name = tokens[1];
		
		// Read the number of atoms in the molecule
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		if(N2 != stoi(tokens[1])){
			
			//Throw exception if the number of atoms don't match
			cout << "The number of atoms N2 = " << N2 << " must match the number of atoms in the molecule Nm2 = " << stoi(tokens[1]) << endl;
			throw 1;
			
		}
		
		// Set the reading precision to the maximum
		myFile.precision(std::numeric_limits<long double>::max_digits10);
		
		// Read the radius of the bounding sphere
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 2);
		
		boundR2 = stold(tokens[1]);
		if(boundR2 < 0.0L) boundR2 = -boundR2;
		
		// Read center of mass
		getline(myFile, line);
		tokens = tokenize_string(line, ',', 3);
		
		for(i = 1; i <= 2; i++) Xcm2i[i-1] = stold(tokens[i]);
		
		// Read the orientation axis
		for(i = 0; i < 2; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 3);
			
			for(j = 1; j <= 2; j++) Xor2i[i][j-1] = stold(tokens[j]);
			
		}
		
		// Read the positions of the atoms in the molecule and their radii
		for(i = 0; i < N2; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 4);
			
			for(j = 1; j <= 2; j++) X2i[i][j-1] = stold(tokens[j]);
			radii2[i] = stold(tokens[3]);
			
		}
		
		// Get the blank line
		getline(myFile, line);
		
		// Read the diffusion tensor
		for(i = 0; i < 3; i++){
			
			getline(myFile, line);
			tokens = tokenize_string(line, ',', 3);
			
			D2[i] = stold(tokens[i]);
			
		}
		
		// Close the file
		myFile.close();
		
	}
	catch(...){
		
		cout << "Failed to read the file for molecule 2. Check the file and try again." << endl;
		exit(EXIT_FAILURE);
		
	}
	
	// Set the molecules to the initial position
	copy_positions(RESETINIT, 4);
	
	// Check the particles are on the correct side
	if(Xcm1[0] < Xcm2[0]){
		
		cout << "Change the sides of particles, the x-position of the first molecule has to be " << endl;
		cout << "greater than the x-position of the second molecule." << endl;
		cout << "x-coordinate of the center of mass of molecule 1: " << Xcm1[0] << endl;
		cout << "x-coordinate of the center of mass of molecule 2: " << Xcm2[0] << endl;
		exit(EXIT_FAILURE);
		
	}
	
}

/*
 
 Sets up molecules.
 
 PARAMETERS:
 
 RETURN:
 
*/
void setup_mols_trial(){
	
	//Auxiliary variables
	int i;
	bool cond;
	long double theta1, theta2, disp;
	
	// Reset the molecules to the initial configuration
	copy_positions(RESETINIT, 4);
	
	// --------------------------------------------------
	// Setup molecule 1
	// --------------------------------------------------
	
	// Rotate the molecules to their initial position
	while(true){
		
		if(N1 <= 1) break;
		
		// Keep the initial positions just in case
		copy_positions(COPYTOTEMP, M1_NUM);
		
		// Choose a random number to get the rotation angle for the first molecule
		theta1 =  2.0L * PI * rand_real1(generator);
		
		for(i = 0; i < N1; i++){
			
			translate_vec(&X1[i][0], -Xcm1[0], -Xcm1[1]);
			rotate_vec(&X1[i][0], theta1);
			translate_vec(&X1[i][0], Xcm1[0], Xcm1[1]);
			
		}
		
		// Evaluate the continuing conditions
		cond = validate_intersect(M1_NUM);
		cond = (cond || validate_intersect(M12_NUM));
		
		// If the molecules don't intersect, continue
		if(!cond){
			
			// Validate position probabilirty
			cond = (rand_real1(generator) < get_position_prob(M1_NUM));
			
			if(cond){
				
				// Rotate the orientation vectors by the same amount
				rotate_vec(&Xor1[0][0], theta1);
				rotate_vec(&Xor1[1][0], theta1);
				break;
				
			}
			
		}
		
		// The molecules intersect with each other or the walls, go to the previous state and try again
		copy_positions(COPYFROMTEMP, M1_NUM);
		
	}
	
	// Set the initial vertical position of molecule 1
	do{
		
		// Keep the initial positions just in case
		copy_positions(COPYTOTEMP, M1_NUM);
		
		// Get the displacement direction and the direction of movement
		if(rand_real1(generator) < 0.5L) disp = get_min_y_pos(X1, radii1, N1);
		else disp = get_max_y_pos(X1, radii1, N1);
		
		// Randomize the displacement
		disp = rand_real1(generator) * disp;
		
		// Make the displacement random
		for(i = 0; i < N1; i++) translate_vec(&X1[i][0], 0.0L, disp);
		
		// Evaluate the continuing conditions
		cond = validate_intersect(M1_NUM);
		cond = (cond || validate_intersect(M12_NUM));
		
		// If the molecules don't intersect, continue
		if(!cond){
			
			translate_vec(Xcm1, 0.0L, disp);         
			break;
			
		}
		
		// The molecules intersect with each other or the walls, go to the previous state and try again
		copy_positions(COPYFROMTEMP, M1_NUM);
		
	}while(true);
	
	// --------------------------------------------------
	// Setup molecule 2
	// --------------------------------------------------
	
	// Rotate the molecules to their initial position
	while(true){
		
		if(N2 <= 1) break;
		
		// Keep the initial positions just in case
		copy_positions(COPYTOTEMP, M2_NUM);
		
		// Choose a random number to get the rotation angle for the second molecule
		theta2 =  2.0L * PI * rand_real1(generator);
		
		for(i = 0; i < N2; i++){
			
			translate_vec(&X2[i][0], -Xcm2[0],-Xcm2[1]);
			rotate_vec(&X2[i][0], theta2);
			translate_vec(&X2[i][0], Xcm2[0], Xcm2[1]);
			
		}
		
		// Evaluate the continuing conditions
		cond = validate_intersect(M2_NUM);
		cond = (cond || validate_intersect(M12_NUM));
		
		// If the molecules don't intersect, continue
		if(!cond){
			
			cond = (rand_real1(generator) < get_position_prob(M2_NUM));
			
			if(cond){
				
				// Rotate the orientation vectors by the same amount
				rotate_vec(&Xor2[0][0], theta2);
				rotate_vec(&Xor2[1][0], theta2);
				break;
				
			}
			
		}
		
		// The molecules intersect with each other or the walls, go to the previous state and try again
		copy_positions(COPYFROMTEMP, M2_NUM);
		
	}
	
	// Set the initial vertical position of the molecules
	do{
		
		// Keep the initial positions just in case
		copy_positions(COPYTOTEMP, M2_NUM);
		
		if(rand_real1(generator) < 0.5L) disp = get_min_y_pos(X2, radii2, N2);
		else disp = get_max_y_pos(X2, radii2, N2);
		
		// Randomize the displacement
		disp = rand_real1(generator) * disp;
		
		// Make the displacement random
		for(i = 0; i < N2; i++) translate_vec(&X2[i][0], 0.0L, disp);
		
		// Evaluate the continuing conditions
		cond = validate_intersect(M2_NUM);
		cond = (cond || validate_intersect(M12_NUM));
		
		// If the molecules don't intersect, continue
		if(!cond){
			
			translate_vec(Xcm2, 0.0L, disp);
			break;
			
		}
		
		// The molecules intersect with each other or the walls, go to the previous state and try again
		copy_positions(COPYFROMTEMP, M2_NUM);
		
	}while(true);
	
}

/*
 
 Sets up a simulation, that is, resets all needed variables to their initial state.
 
 PARAMETERS:
 
	- long double poreW : The width of the pore.
	- bool toZero : If the pass and fail counters should be set to zero.
 
 RETURN:
 
*/
void setup_trial(long double poreW, bool toZero){
	
	if(!loadSim){
		
		if(toZero){
			
			// First attempt
			firstTime = true;
			
			// Set the passes and fails counters to zero.
			fails = 0;
			passes = 0;
			
			// Set the save counter to zero
			saveCnt = SAVEAFT;
			
			// Initialize the percentage counter
			perc = 1.0L;
			sim_percentage(-1);
			
			// Set the pore top and bottom coordinates
			poreTop = 0.5L * poreW;
			poreBot = - 0.5L * poreW;
			
			// Set the pore width
			poreWidth = poreW;
			
			// Setup the tolerance limits
			TOLLIM1 = get_tollim(M1_NUM);
			TOLLIM2 = get_tollim(M2_NUM);
			
			// Get the maximum rotation angles
			MAXANGL1 = get_angle_max(M1_NUM);
			MAXANGL2 = get_angle_max(M2_NUM);
			
			// Write the time step
			write_prob_results(WRITE_FILE, -1.0L);
			
		}
		
		setup_mols(!TO_ZERO);
		
		successPass = -(Xcm1[0] - Xcm2[0]);
		failPass = 2.0L * (Xcm1[0] - Xcm2[0]);
		
	}
	else{
		
		// Initialize the percentage counter
		perc = 1.0L;
		sim_percentage(-1);
		sim_percentage(passes + fails);
		
		// Set the pore top and bottom coordinates
		poreTop = 0.5L * poreW;
		poreBot = - 0.5L * poreW;
		
		// Set the pore width
		poreWidth = poreW;
		
		// Setup the tolerance limits
		TOLLIM1 = get_tollim(M1_NUM);
		TOLLIM2 = get_tollim(M2_NUM);
		
		// Get the maximum rotation angles
		MAXANGL1 = get_angle_max(M1_NUM);
		MAXANGL2 = get_angle_max(M2_NUM);
		
		// Write the time step
		write_prob_results(WRITE_FILE, -1.0L);
		
		// File must only be loaded once
		loadSim = false;
		
	}
	
}

//------------------------------------------------------
//   TRANSLATE MOLECULES RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Translates the given molecule.
 
 PARAMETERS:
 
	- int molNum : The id of the molecule to rotate.
 
 RETURN:
 
*/
void translate_mol(int molNum){
	
	//Auxiliary variables
	int i;
	long double dx, dy, trans[2];
	
	// Translate molecule 1
	if(molNum == M1_NUM || molNum == M12_NUM){
		
		// Get a translation in the x and y direction
		dx = get_disp(D1[0]);
		dy = get_disp(D1[1]);
		
		// Get the vector translation
		trans[0] = dx * Xor1[0][0] + dy * Xor1[1][0];
		trans[1] = dx * Xor1[0][1] + dy * Xor1[1][1];
		
		// Translate the molecule
		for(i = 0; i < N1; i++) translate_vec(&X1[i][0], trans[0], trans[1]);
		
		// Translate the center of mass
		translate_vec(Xcm1, trans[0], trans[1]);
		
	}
	
	// Translate molecule 2
	if(molNum == M2_NUM || molNum == M12_NUM){
		
		// Get a translation in the x and y direction
		dx = get_disp(D2[0]);
		dy = get_disp(D2[1]);
		
		// Get the vector translation
		trans[0] = dx * Xor2[0][0] + dy * Xor2[1][0];
		trans[1] = dx * Xor2[0][1] + dy * Xor2[1][1];
		
		// Translate the molecule
		for(i = 0; i < N2; i++) translate_vec(&X2[i][0], trans[0], trans[1]);
		
		// Translate the center of mass
		translate_vec(Xcm2, trans[0], trans[1]);
		
	}
	
}

/*
 
 Translates a 2D vector by dx in the x component and dy in the y component.
 
 PARAMETERS:
 
	- long double * vec : Pointer to the first element of a 2 dimensional vector.
	- long double dx : Displacement in the x-direction.
	- long double dy : Displacement in the y-direction.
 
 RETURN:
 
*/
void translate_vec(long double * vec, long double dx, long double dy){
	
	vec[0] += dx;
	vec[1] += dy;
	
}

//------------------------------------------------------
//   VALIDATION RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Validate if two molecules intersect each other or a molecule intersects the pore wall.
 
 PARAMETERS:
 
 RETURN:
 
	- bool cond : If the molecules intersect with each other or the wall, cond is true; otherwise it's false.
 
*/
bool validate_intersect(int molNum){
	
	//Auxiliary variables
	int i, j;
	bool cond;
	long double radMinSep, rx, ry;
	
	// Initially the condition is false
	cond = false;
	
	if(molNum == M1_NUM){
		
		// First check the bounding sphere
		if((2.0L * boundR1) < (poreTop - poreBot)){
			
			// Get the top most part of the bounding sphere in y
			rx = Xcm1[1] + boundR2;
			ry = Xcm1[1] - boundR2;
			
			// If the bounding sphere intersects the pore bottom
			if(rx <= poreTop && ry >= poreBot) return cond;
			
		}
		
		// Not always all circles will have to be examined
		for(i = 0; i < N1; i++){
			
			radMinSep = X1[i][1] + radii1[i];
			cond = (radMinSep > poreTop);
			
			// If it intersects the wall stop
			if(cond) break;
			
			radMinSep = X1[i][1] - radii1[i];
			cond = (radMinSep < poreBot);
			
			// If it intersects the wall stop 
			if(cond) break;
			
		}
		
	}
	else if(molNum == M2_NUM){
		
		// First check the bounding sphere
		if((2.0L * boundR2) < (poreTop - poreBot)){
			
			// Get the top most part of the bounding sphere in y
			rx = Xcm2[1] + boundR2;
			ry = Xcm2[1] - boundR2;
			
			// If the bounding sphere intersects the pore bottom
			if(rx <= poreTop && ry >= poreBot) return cond;
			
		}
		
		// Not always all circles will have to be examined
		for(i = 0; i < N2; i++){
			
			radMinSep = X2[i][1] + radii2[i];
			cond = (radMinSep > poreTop);
			
			// If it intersects the wall stop
			if(cond) break;
			
			radMinSep = X2[i][1] - radii2[i];
			cond = (radMinSep < poreBot);
			
			// If it intersects the wall stop 
			if(cond) break;
			
		}
		
	}
	else if(molNum == M12_NUM){
		
		//First check the bounding spheres
		rx = (Xcm1[0] - Xcm2[0]) * (Xcm1[0] - Xcm2[0]);
		ry = (Xcm1[1] - Xcm2[1]) * (Xcm1[1] - Xcm2[1]);
		
		// The minimum separation between the molecules
		radMinSep = (boundR1 + boundR2) * (boundR1 + boundR2);
		
		// If the bounding spheres doesn't intersect
		if(radMinSep <= (rx + ry)) return cond;
		
		for(i = 0; i < N1 && !cond; i++){
   
			for(j = 0; j < N2 && !cond; j++){
				
				// The radial separation between the molecules
				rx = (X1[i][0] - X2[j][0]) * (X1[i][0] - X2[j][0]) ;
				ry = (X1[i][1] - X2[j][1]) * (X1[i][1] - X2[j][1]);
				
				// The minimum separation between the molecules
				radMinSep = (radii1[i] + radii2[j]) * (radii1[i] + radii2[j]) ;
				
				// Check if the molecules intersect
				cond = (radMinSep > (rx + ry));
				
			}
   
		}
		
	}
	
	return cond;
	
}

//------------------------------------------------------
//  WRITE FILES RELATED SUBROUTINES
//------------------------------------------------------

/*
 
 Writes the basic information of the simulation for the molecules.
 
 PARAMETERS:
 
	- ofstream * fileToWrite : An already open stream to write to a file.
 
 RETURN:
 
*/
void write_file_prelude(ofstream * fileToWrite){
	
	// Auxiliary variables
	int i;
	
	//----------------------------------------
	// For molecule 1
	//----------------------------------------
	
	(*fileToWrite) << endl;
	(*fileToWrite) << "Molecule 1 : " << mol1Name;
	(*fileToWrite) << ",Number of Atoms = " << N1;
	(*fileToWrite) << ",Diffusion Coefficients {Dx;Dy;D_ang} = {" << D1[0] << ";" << D1[1] << ";" << D1[2] << "}";
	(*fileToWrite) << ",Radii = {";
	for(i = 0; i < N1; i++){
		
		if(i != 0) (*fileToWrite) << ";";
		(*fileToWrite) << radii1[i];
		
	}
	(*fileToWrite) << "}";
	
	(*fileToWrite) << endl;
	
	//----------------------------------------
	// For molecule 2
	//----------------------------------------
	
	(*fileToWrite) << "Molecule 2 : " << mol2Name;
	(*fileToWrite) << ",Number of Atoms = " << N2;
	(*fileToWrite) << ",Diffusion Coefficients {Dx;Dy;D_ang} = {" << D2[0] << ";" << D2[1] << ";" << D2[2] << "}";
	(*fileToWrite) << ",Radii = {";
	for(i = 0; i < N2; i++){
		
		if(i != 0) (*fileToWrite) << ";";
		(*fileToWrite) << radii2[i];
		
	}
	(*fileToWrite) << "}";
	
	(*fileToWrite) << endl;
	
}

/*
 
 Writes the current configuration of the pore.
 
 PARAMETERS:
 
	- int opt : The option for the file operation.
 
 RETURN:
 
*/
void write_mol_config(int opt){
	
	//Auxiliary variables
	int i;
	ofstream myFile;
	
	if(opt == OPEN_FILE || opt == OPEN_FILE_OVER){
		
		// Open the file
		myFile.open(FILE_SAVE_CONF);
		
		myFile.precision(std::numeric_limits<long double>::max_digits10);
		myFile << poreTop << "," << poreBot << "," << boundR1 << "," << boundR2 << endl;
		
		myFile << radii1[0];
		for(i = 1; i < N1; i++) myFile << "," << radii1[i];
		myFile << endl;
		
		myFile << radii2[0];
		for(i = 1; i < N2; i++) myFile << "," << radii2[i];
		myFile << endl;
		
		// Close the file
		myFile.close();
		
	}
	else if (opt == WRITE_FILE){
		
		// Open the file
		myFile.open(FILE_SAVE_CONF, std::ofstream::out | std::ofstream::app);
		
		// Molecule 1 info
		for(i = 0; i < N1; i++) myFile << X1[i][0] << "," << X1[i][1] << ",";
		
		myFile << "-," << Xor1[0][0] << "," << Xor1[0][1]; 
		myFile << ",-," << Xor1[1][0] << "," << Xor1[1][1];
		myFile << ",-," << Xcm1[0] << "," << Xcm1[1] << endl;
		
		// Molecule 2 info
		for(i = 0; i < N2; i++) myFile << X2[i][0] << "," << X2[i][1] << ",";
		
		myFile << "-," << Xor2[0][0] << "," << Xor2[0][1];
		myFile << ",-," << Xor2[1][0] << "," << Xor2[1][1];
		myFile << ",-," << Xcm2[0] << "," << Xcm2[1] << endl;
		
		// Close the file
		myFile.close();
		
	}
	else if (opt == NEW_LINE_FILE){
		
		myFile.open(FILE_SAVE_CONF, std::ofstream::out | std::ofstream::app);
		myFile << endl;
		myFile.close();
		
	}
	
}

/*
 
 Writes the results for a particular time step of the simulation.
 
 PARAMETERS:
 
	- int opt : The option to manipulate the file.
	- long double results_r : The results for a specific time step.
 
 RETURN:
 
*/
void write_prob_results(int opt, long double results_r){
	
	// Auxiliary variables
	int i;
	ofstream myFile;
	
	if(opt == OPEN_FILE || opt == OPEN_FILE_OVER){
		
		// Open the file
		myFile.open(FILE_SAVE_NAME, std::ofstream::out | std::ofstream::app);
		
		// Write the prelude
		write_file_prelude(&myFile);
		
		//Close the file
		myFile.close();
		
	}
	else if (opt == WRITE_FILE){
		
		// Open the file
		myFile.open(FILE_SAVE_NAME, std::ofstream::out | std::ofstream::app);
		
		if(results_r < 0.0L){
			
			// Write the header section for the simulation
			myFile << "Time step:," << timesAndWidths[0][0] << ",Pore Width:," << timesAndWidths[0][1] << endl;
			
			if(loadSim && (passes+fails) > 0) myFile << (fails + passes) << ",";
			
			for(i = saveCnt; i < MAXCNTR; i += SAVEAFT){
				
				if(i != saveCnt) myFile << ",";
				myFile << i;  
				
			}
			myFile << "," << MAXCNTR << endl;
			
			// Write the result
			myFile.precision(std::numeric_limits<long double>::max_digits10);
			if(loadSim && (passes+fails) > 0) myFile << ((long double) passes)/((long double) (fails + passes)) << ",";   
			
		}
		else{
			
			// Write the result
			myFile.precision(std::numeric_limits<long double>::max_digits10);
			
			if(!firstTime) myFile << ",";
			myFile << results_r;
			
		}
		
		// Close the file
		myFile.close();
		
	}
	else if (opt == NEW_LINE_FILE){
		
		myFile.open(FILE_SAVE_NAME, std::ofstream::out | std::ofstream::app);
		myFile << endl;
		myFile.close();
		
	}
 
}

//------------------------------------------------------
//  OTHER SUBROUTINES
//------------------------------------------------------

/*
 
 Raises the integer number num1 to the power of an integer number num2, num2 is positive, otherwise it returns 1.
 
 PARAMETERS:
 
	- int num1 : The integer number to be raised to the n^th power.
	- int num2 : The number to which to raise num1, num2 >= 0.
 
 RETURN:
 
	- int num1 ^ num2 : Number num1 raised to the number num2.
 
*/
int int_pow(int num1, int num2){
	
	if(num2 <= 0) return 1;
	
	if(num1 == 0) return 0;
	
	return num1 * int_pow(num1, num2 - 1);
	
}

/*
 
 Raises the number long double num1 to the power of an integer number num2, num2 is positive, otherwise it returns 1.
 
 PARAMETERS:
 
	- long double num1 : The integer number to be raised to the n^th power.
	- int num2 : The number to which to raise num1, num2 >= 0.
 
 RETURN:
 
	- long double num1 ^ num2 : Number num1 raised to the number num2.
 
*/
long double int_pow_L(long double num1, int num2){
	
	if(num2 == 0) return 1.0L;
	
	if(num1 == 0.0L) return 0.0L;
	
	if(num2 < 0) return (1.0L/num1) * int_pow_L(num1, num2 + 1); 
	
	return num1 * int_pow_L(num1, num2 - 1);
	
}

/*
 
 Gets the maximum value in an array of long double values.
 
 PARAMETERS:
 
	- long double [] arrayE : Array that contains a set of values.
	- int sizeA : The length of the array.
 
 RETURN:
 
	- long double maxArray : The biggest value in an array.
 
*/
long double max_of_array(long double arrayE[], int sizeA){
	
	// Auxiliary variables
	long double maxArray;
	
	
	// Set the first element as the maximum
	maxArray = arrayE[0];
	
	// Get the maximum element in the array
	for(int i = 1; i < sizeA; i++){
		
		// Compare the elements
		if(maxArray < arrayE[i]) maxArray = arrayE[i];
		
	}  
	
	// Return the maximum
	return maxArray; 
	
}

/*
 
 Gets the minimum value in an array of long double values.
 
 PARAMETERS:
 
	- long double [] arrayE : Array that contains a set of values.
	- int sizeA : The length of the array.
 
 RETURN:
 
	- long double minArray : The smalles value in an array.
 
*/
long double min_of_array(long double arrayE[], int sizeA){
	
	// Auxiliary variables
	long double minArray;
	
	
	// Set the first element as the maximum
	minArray = arrayE[0];
	
	// Get the maximum element in the array
	for(int i = 1; i < sizeA; i++){
		
		// Compare the elements
		if(minArray > arrayE[i]) minArray = arrayE[i];
		
	}  
	
	// Return the maximum
	return minArray; 
	
}


/*
 
 Prints the information of the given molecule or molecules.
 
 PARAMETERS:
 
	- int molNum : The number of the molecule for which to print the information.
 
 RETURN:
 
*/
void print_mol_info(int molNum){
	
	// Auxiliary variables
	int i;
	
	if(molNum == M1_NUM || molNum == M12_NUM){
		
		cout << endl << "-----------------------------------" << endl;
		cout << "- Molecule 1 info:";
		cout << endl << "-----------------------------------" << endl << endl;
		
		cout << "Bounding radius 1: " << boundR1 << endl;
		cout << "Center of mass Xcm1 -> (x,y): (" << Xcm1[0] << ", " << Xcm1[1] << ")" << endl;
		if(N1 > 1) cout << "Orientation vector x' X1or[0] -> (x,y): (" << Xor1[0][0] << ", " << Xor1[0][1] << ")" << endl;
		if(N1 > 1) cout << "Orientation vector y' X1or[1] -> (x,y): (" << Xor1[1][0] << ", " << Xor1[1][1] << ")" << endl;
		for(i = 0; i < N1; i++) cout << (i+1) << ". Atom position X1[" << i <<"] -> (x,y)  and radius of atom r: (" << X1[i][0] << ", " << X1[i][1] << ")" << ", r = " << radii1[i] << endl;
		cout << "Diffusion tensor D1 = (Dx, Dy, Dtheta): (" << D1[0] << ", " << D1[1] << ", " << D1[2] << ")" << endl;
		
		cout << endl << "end of information" << endl;
		cout << endl << "-----------------------------------" << endl;
		
	}
	
	if(molNum == M2_NUM || molNum == M12_NUM){
		
		cout << endl << "-----------------------------------" << endl;
		cout << "- Molecule 2 info:";
		cout << endl << "-----------------------------------" << endl << endl;
		
		cout << "Bounding radius 2: " << boundR2 << endl;
		cout << "Center of mass Xcm2 -> (x,y): (" << Xcm2[0] << ", " << Xcm2[1] << ")" << endl;
		if(N2 > 1) cout << "Orientation vector x' X2or[0] -> (x,y): (" << Xor2[0][0] << ", " << Xor2[0][1] << ")" << endl;
		if(N2 > 1) cout << "Orientation vector y' X2or[1] -> (x,y): (" << Xor2[1][0] << ", " << Xor2[1][1] << ")" << endl;
		for(i = 0; i < N2; i++) cout << (i+1) << ". Atom position X2[" << i <<"] -> (x,y)  and radius of atom r : (" << X2[i][0] << ", " << X2[i][1] << ")" << ", r = " << radii2[i] << endl;
		cout << "Diffusion tensor D2 = (Dx, Dy, Dtheta): (" << D2[0] << ", " << D2[1] << ", " << D2[2] << ")" << endl;
		
		cout << endl << "end of information" << endl;
		cout << endl << "-----------------------------------" << endl;
		
	}
	
}

/*
 
 Raises the number long double num1 to the power of an integer number num2, num2 is positive, otherwise it returns 1.
 
 PARAMETERS:
 
	- long double prog : The progress made. If the progress is less than zero, the counter is started.
 
 RETURN:
 
*/
void sim_percentage(int prog){
	
	// Auxiliary variables
	int i;
	long double cPerc;
	
	if(prog < 0){
		
		cout << "\n\t";
		for(i = 1; i <= 10; i++) cout << "X";
		for(i = 1; i <= 10; i++) cout << "O";
		for(i = 1; i <= 10; i++) cout << "X";
		for(i = 1; i <= 10; i++) cout << "O";
		for(i = 1; i <= 10; i++) cout << "X";
		for(i = 1; i <= 10; i++) cout << "O";
		for(i = 1; i <= 10; i++) cout << "X";
		for(i = 1; i <= 10; i++) cout << "O";
		for(i = 1; i <= 10; i++) cout << "X";
		for(i = 1; i <= 10; i++) cout << "O";
		cout << endl;
		
	}
	else{
		
		cPerc = 100.0L *  ((long double) prog) / ((long double) MAXCNTR);
		
		while(cPerc >= perc){
			
			if(perc == 1.0L) cout << "\t";
			perc += 1.0L;
			
			cout << "+";
			if(cPerc == 100.0L) cout << endl << endl;
			cout.flush();
			
		}
		
		// Turn the number of passes and fails into a long double number
		if(saveCnt == (passes + fails)){
			
			while(saveCnt <= (passes + fails)) saveCnt += SAVEAFT;
			if(saveCnt > MAXCNTR) saveCnt = MAXCNTR; 
			if(SAVE_FILE) fr_save_pr();
			write_prob_results(WRITE_FILE, ((long double) passes) / prog);
			firstTime = false;
			
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
		
		if((int) cont.size() != len){
			
			cout << "The length of the vector should be " << len << ", not " << cont.size() << endl;
			cout << "This is the string:\n\t" << st << endl;
			throw 1;
			
		}
		
	}
	
	// Return the container with the tokens
	return cont;
	
}
