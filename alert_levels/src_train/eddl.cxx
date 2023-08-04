#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <math.h>
#include <sys/stat.h> 
#include <sys/time.h>
#include "eddl/apis/eddl.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/serialization/onnx/onnx.pb.h"

#define GPUS_PER_NODE 1

// Modify these mininum and maximum values accordingly
#define min_lon    -8.2
#define max_lon    -6.7
#define min_lat    36.0
#define max_lat    36.9
#define min_depth  5.0
#define max_depth  15.0
#define min_length 25.0
#define max_length 85.0
#define min_width  20.0
#define max_width  70.0
#define min_strike -110.0
#define max_strike -90.0
#define min_dip    20.0
#define max_dip    40.0
#define min_rake   80.0
#define max_rake   100.0
#define min_slip   2.0
#define max_slip   8.0

// Loss function used in early stopping. 1: categorical cross entropy; 2: accuracy.
#define LOSS_ES    1

// Initialization flags of the data sets  (do not change)
#define EXACT      0
#define PERCENTAGE 1

using namespace std;
using namespace eddl;

/**********************/
/* Guardado de modelo */
/**********************/

void generarCadenaAleatoria(char *cadena, int tam)
{
	char v[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	struct timeval tv;
	int seed, i, ind;

	gettimeofday(&tv, NULL);
	seed = tv.tv_usec;

	mt19937 generator(seed);
	uniform_int_distribution<> dis(0,61);
	for (i=0; i<tam; i++) {
		ind = dis(generator);
		cadena[i] = v[ind];
	}
	cadena[tam] = '\0';
}

void guardarMetadatosEnFicheroONNX(string path, int epochs, int batch_size, int patience_es, float lr_inicial,
		int reducir_lr,	int patience_lr, float lr_final)
{
	char cadena[16];
	onnx::ModelProto model;
	onnx::StringStringEntryProto *prop;
	ifstream in(path, ios_base::binary);
	model.ParseFromIstream(&in);
	in.close();

	// Añadir metadatos al modelo
	prop = model.add_metadata_props();
	sprintf(cadena, "%d", epochs);
	prop->set_key("epochs");
	prop->set_value(cadena);
	prop = model.add_metadata_props();
	sprintf(cadena, "%d", batch_size);
	prop->set_key("batch_size");
	prop->set_value(cadena);
	prop = model.add_metadata_props();
	sprintf(cadena, "%d", patience_es);
	prop->set_key("patience_es");
	prop->set_value(cadena);
	prop = model.add_metadata_props();
	prop->set_key("optimizer");
	prop->set_value("Adam");
	prop = model.add_metadata_props();
	sprintf(cadena, "%e", lr_inicial);
	prop->set_key("initial_lr");
	prop->set_value(cadena);
	prop = model.add_metadata_props();
	sprintf(cadena, "%d", reducir_lr);
	prop->set_key("reduce_lr");
	prop->set_value(cadena);
	if (reducir_lr != 0) {
		prop = model.add_metadata_props();
		sprintf(cadena, "%d", patience_lr);
		prop->set_key("patience_lr");
		prop->set_value(cadena);
		prop = model.add_metadata_props();
		sprintf(cadena, "%e", lr_final);
		prop->set_key("final_lr");
		prop->set_value(cadena);
	}

	// Crear el stream y guardar los metadatos en el fichero ONNX
	fstream ofs(path, ios::out | ios::trunc | ios::binary);
	if (! model.SerializeToOstream(&ofs)) {
		cerr << "Error: Failed to add metadata to ONNX file" << endl;
	}
	ofs.close();
}

/********************/
/* Lectura de datos */
/********************/

// Devuelve true si existe el fichero, false en otro caso
bool existeFichero(string fichero) 
{
	struct stat stFichInfo;
	bool existe;
	int intStat;

	// Obtenemos los atributos del fichero
	intStat = stat(fichero.c_str(), &stFichInfo);
	if (intStat == 0) {
		// Hemos obtenido los atributos del fichero. Por tanto, el fichero existe.
		existe = true;
	}
	else {
		// No hemos obtenido los atributos del fichero. Notar que esto puede
		// significar que no tenemos permiso para acceder al fichero. Para
		// hacer esta comprobación, comprobar los valores de intStat.
		existe = false;
	}

	return existe;
}

template <class T>
void obtenerSiguienteDato(ifstream &fich, T &dato)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> dato;
}

void obtenerNumeroElementosConjuntos(ifstream &fich, int &num_train, int &num_val, int &num_test)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");  
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> num_train >> num_val >> num_test;
}

void obtenerPorcentajeConjuntos(ifstream &fich, float &porc_train, float &porc_val, float &porc_test)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");  
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> porc_train >> porc_val >> porc_test;
}

// Devuelve 0 si todo ha ido bien, 1 si ha habido algún error
int cargarDatosProblema(string fich_ent, string &fich_datosX, string &fich_datosY, string &directorio, int *flag_sets,
			int *num_train, int *num_val, int *num_test, float *porc_train, float *porc_val, float *porc_test, int *num_capas,
			int *num_unidades, int *max_epochs, int *batch_size, int *patience_es, float *lr_inicial, int *reducir_lr,
			int *patience_lr, float *lr_final, string &fich_onnx, int id_hebra)
{
	int i;
	double val;

	// Ponemos en directorio el directorio donde están los ficheros de datos
	i = fich_ent.find_last_of("/");
	if (i > -1) {
		// Directorios indicados con '/' (S.O. distinto de windows)
		directorio = fich_ent.substr(0,i)+"/";
	}
	else {
		i = fich_ent.find_last_of("\\");
		if (i > -1) {
			// Directorios indicados con '\' (S.O. Windows)
			directorio = fich_ent.substr(0,i)+"\\";
		}
		else {
			// No se ha especificado ningún directorio para los ficheros de datos
			directorio = "";
		}
	}

	ifstream fich(fich_ent.c_str());
	obtenerSiguienteDato<string>(fich, fich_datosX);
	if (! existeFichero(directorio+fich_datosX)) {
		if (id_hebra == 0)
			cerr << "Error: File '" << directorio+fich_datosX << "' not found" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<string>(fich, fich_datosY);
	if (! existeFichero(directorio+fich_datosY)) {
		if (id_hebra == 0)
			cerr << "Error: File '" << directorio+fich_datosY << "' not found" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *flag_sets);
	if (*flag_sets == EXACT) {
		// Se da el número de elementos de train, val y test
		obtenerNumeroElementosConjuntos(fich, *num_train, *num_val, *num_test);
	}
	else {
		// Se da el porcentaje de elementos de train, val y test
		obtenerPorcentajeConjuntos(fich, *porc_train, *porc_val, *porc_test);
		val = (*porc_train) + (*porc_val) + (*porc_test);
		if (fabs(val-100.0) > 1e-6) {
			if (id_hebra == 0)
				cerr << "Error: The sum of train, validation and test percentages should be 100" << endl;
			fich.close();
			return 1;
		}
	}
	obtenerSiguienteDato<int>(fich, *num_capas);
	if (*num_capas <= 0) {
		if (id_hebra == 0)
			cerr << "Error: The number of layers should be > 0" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *num_unidades);
	if (*num_unidades <= 0) {
		if (id_hebra == 0)
			cerr << "Error: The number of units per layer should be > 0" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *max_epochs);
	if (*max_epochs <= 0) {
		if (id_hebra == 0)
			cerr << "Error: The maximum number of epochs should be > 0" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *batch_size);
	if (*batch_size <= 0) {
		if (id_hebra == 0)
			cerr << "Error: The batch size should be > 0" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *patience_es);
	obtenerSiguienteDato<float>(fich, *lr_inicial);
	obtenerSiguienteDato<int>(fich, *reducir_lr);
	if (*reducir_lr == 1) {
		obtenerSiguienteDato<int>(fich, *patience_lr);
		obtenerSiguienteDato<float>(fich, *lr_final);
	}
	obtenerSiguienteDato<string>(fich, fich_onnx);
	fich.close();

	return 0;
}

void mostrarDatosProblema(int flag_sets, int num_ejemplos, int num_train, int num_val, int num_test, float porc_train,
		float porc_val, float porc_test, int num_capas, int num_unidades, int max_epochs, int batch_size, int patience_es,
		float lr_inicial, int reducir_lr, int patience_lr, float lr_final)
{
	cout << endl << "Problem data" << endl;
	cout << "Number of samples: " << num_ejemplos << endl;
	if (flag_sets == EXACT) {
		cout << "Initialization of the train, validation and test sets: Exact" << endl;
		cout << "  Train: " << num_train << ", validation: " << num_val << ", test: " << num_test << endl;
	}
	else {
		cout << "Initialization of the train, validation and test sets: Percentage" << endl;
		cout << "  Train: " << porc_train << ", validation: " << porc_val << ", test: " << porc_test << endl;
	}
	cout << "Number of hidden layers: " << num_capas << endl;
	cout << "Number of units per hidden layer: " << num_unidades << endl;
	cout << "Maximum number of epochs: " << max_epochs << endl;
	cout << "Batch size: " << batch_size << endl;
	cout << "Patience for the early stopping: " << patience_es << endl;
	cout << "Initial learning rate: " << lr_inicial << endl;
	if (reducir_lr == 0) {
		cout << "Reduce learning rate on plateau: no" << endl;
	}
	else {
		cout << "Reduce learning rate on plateau: yes" << endl;
		cout << "  Patience: " << patience_lr << endl;
		cout << "  Final lerning rate: " << lr_final << endl;
	}
	cout << endl;
}

/*****************/
/* Deep learning */
/*****************/

// Imprime la matriz de confusión, precision y recall. Los tensores de entrada tienen una columna
// con la clase correspondiente (no son probabilidades)
void matrizDeConfusion(int num_clases, Tensor *y_true, Tensor *y_pred)
{
    Tensor *aux, *ciertas, *predichas;
    std::pair<unsigned int *, int> res;
    int *matriz = new int[num_clases*num_clases];
    int num_aciertos;
    double suma;
    int i, j;

    // Obtenemos la matriz de confusión
    fprintf(stdout, "Matriz de confusion:\n");
    for (i=0; i<num_clases; i++) {
        aux = Tensor::full_like(y_true, (float) i);
        // ciertas[k]: 1 si el ejemplo k tiene clase cierta i, 0 en otro caso
        ciertas = y_true->isclose(aux);
        for (j=0; j<num_clases; j++) {
            aux = Tensor::full_like(y_pred, (float) j);
            // predichas[k]: 1 si el ejemplo k se ha predicho de clase j, 0 en otro caso
            predichas = y_pred->isclose(aux);
            res = ciertas->logical_and(predichas)->_nonzero();
            num_aciertos = res.second;
            matriz[i*num_clases+j] = num_aciertos;
            fprintf(stdout, "%8d", num_aciertos);
        }
        fprintf(stdout, "\n");
    }

    // Obtenemos precision y recall
    fprintf(stdout, "\nPrecision (%%): ");
    for (i=0; i<num_clases; i++) {
        suma = 0.0;
        for (j=0; j<num_clases; j++)
            suma += (double) matriz[j*num_clases+i];
        fprintf(stdout, "%.2f", matriz[i*num_clases+i]*100.0/suma);
        if (i < num_clases-1)
            fprintf(stdout, ", ");

    }

    fprintf(stdout, "\nRecall (%%):    ");
    for (i=0; i<num_clases; i++) {
        suma = 0.0;
        for (j=0; j<num_clases; j++)
            suma += (double) matriz[i*num_clases+j];
        fprintf(stdout, "%.2f", matriz[i*num_clases+i]*100.0/suma);
        if (i < num_clases-1)
            fprintf(stdout, ", ");
    }
    fprintf(stdout, "\n");

    delete [] matriz;
}

void build_model(model *net, float lr_inicial, bool use_cpu, int seed, int num_clases, int num_capas, int num_unidades, int id_hebra)
{
	compserv cs = nullptr;
	vector<int> gpus(GPUS_PER_NODE, 0);
	int i, id_gpu;

    layer in = Input({9});
    layer l = in;
    for (i=0; i<num_capas; i++) {
        l = Tanh(GlorotUniform(Dense(l, num_unidades), seed));
    }
    layer out = Softmax(Dense(l, num_clases), -1);
    *net = Model({in}, {out});
    (*net)->verbosity_level = 0;

	// dot de graphviz debe estar instalado
//	plot(*net, "model.pdf");

	if (use_cpu) {
		cs = CS_CPU(1);
	} else {
		id_gpu = id_hebra%GPUS_PER_NODE;
		gpus[id_gpu] = 1;
		cs = CS_GPU(gpus);
	}

	build(*net, adam(lr_inicial), {"categorical_cross_entropy"}, {"accuracy"}, cs);

/*vector<vtensor> pesos;
pesos = get_parameters(*net, true);
for(i=0; i<pesos.size(); i++) {
	vtensor p = pesos[i];
	for(int j=0; j<p.size(); j++) {
		fprintf(stdout,"capa %d, vector %d\n", i+1, j+1);
		p[j]->print();
	}
}*/

//	summary(*net);
}

void cargar_datos(string directorio, string nombre_fichX, string nombre_fichY, int flag_sets, int *num_ejemplos,
			int num_train, int num_val, int num_test, float porc_train, float porc_val, float porc_test, int num_clases,
			int seed, Tensor **x_train, Tensor **y_train, Tensor **x_val, Tensor **y_val, Tensor **x_test, Tensor **y_test)
{
	ifstream fichX(directorio+nombre_fichX);
	ifstream fichY(directorio+nombre_fichY);
	double val;
	int i, j, ind;
	char ind_fila[16];
	const char *ind_col[9] = {"0","1","2","3","4","5","6","7","8"};
	int inc_reserva = 10000;
	int num_elems_reserva;
	float *X = (float *) malloc(inc_reserva*9*sizeof(float));
	float *Y = (float *) malloc(inc_reserva*num_clases*sizeof(float));
	float factor_lon = 1.0/(max_lon-min_lon);
	float factor_lat = 1.0/(max_lat-min_lat);
	float factor_depth = 1.0/(max_depth-min_depth);
	float factor_length = 1.0/(max_length-min_length);
	float factor_width = 1.0/(max_width-min_width);
	float factor_strike = 1.0/(max_strike-min_strike);
	float factor_dip = 1.0/(max_dip-min_dip);
	float factor_rake = 1.0/(max_rake-min_rake);
	float factor_slip = 1.0/(max_slip-min_slip);

	num_elems_reserva = inc_reserva;
	i = 0;
	while (fichX >> val) {
		if (i == num_elems_reserva) {
			num_elems_reserva += inc_reserva;
			X = (float *) realloc(X, num_elems_reserva*9*sizeof(float));
			Y = (float *) realloc(Y, num_elems_reserva*num_clases*sizeof(float));
		}
		// lon
		val = (val-min_lon)*factor_lon;
		X[9*i] = val;
		// lat
		fichX >> val;
		val = (val-min_lat)*factor_lat;
		X[9*i + 1] = val;
		// depth
		fichX >> val;
		val = (val-min_depth)*factor_depth;
		X[9*i + 2] = val;
		// fault length
		fichX >> val;
		val = (val-min_length)*factor_length;
		X[9*i + 3] = val;
		// fault width
		fichX >> val;
		val = (val-min_width)*factor_width;
		X[9*i + 4] = val;
		// strike
		fichX >> val;
		val = (val-min_strike)*factor_strike;
		X[9*i + 5] = val;
		// dip
		fichX >> val;
		val = (val-min_dip)*factor_dip;
		X[9*i + 6] = val;
		// rake
		fichX >> val;
		val = (val-min_rake)*factor_rake;
		X[9*i + 7] = val;
		// slip
		fichX >> val;
		val = (val-min_slip)*factor_slip;
		X[9*i + 8] = val;
		i++;
	}
	*num_ejemplos = i;
	for (i=0; i<(*num_ejemplos); i++) {
		for (j=0; j<num_clases; j++) {
			fichY >> val;
			Y[num_clases*i+j] = val;
		}
	}
	fichX.close();
	fichY.close();

	if (flag_sets == PERCENTAGE) {
		// Obtenemos el número de ejemplos de train, val y test a partir de los porcentajes
		num_train = round((*num_ejemplos)*porc_train*0.01);
		num_val   = round((*num_ejemplos)*porc_val*0.01);
		num_test  = round((*num_ejemplos)*porc_test*0.01);
		num_test = (*num_ejemplos)-num_train-num_val;
	}

	*x_train = new Tensor({num_train,9});
	*y_train = new Tensor({num_train,num_clases});
	*x_val = new Tensor({num_val,9});
	*y_val = new Tensor({num_val,num_clases});
	*x_test = new Tensor({num_test,9});
	*y_test = new Tensor({num_test,num_clases});

	// Crear aleatoriamente conjuntos de entrenamiento, validación y test
	vector<int> pos;
	srand(seed);
	for (i=0; i<num_train+num_val+num_test; i++)
		pos.push_back(i);
	random_shuffle(pos.begin(), pos.end());

	for (i=0; i<num_train; i++) {
		ind = pos[i];
		sprintf(ind_fila, "%d", i);
		(*x_train)->set_select({ind_fila,ind_col[0]}, X[ind*9]);
		(*x_train)->set_select({ind_fila,ind_col[1]}, X[ind*9+1]);
		(*x_train)->set_select({ind_fila,ind_col[2]}, X[ind*9+2]);
		(*x_train)->set_select({ind_fila,ind_col[3]}, X[ind*9+3]);
		(*x_train)->set_select({ind_fila,ind_col[4]}, X[ind*9+4]);
		(*x_train)->set_select({ind_fila,ind_col[5]}, X[ind*9+5]);
		(*x_train)->set_select({ind_fila,ind_col[6]}, X[ind*9+6]);
		(*x_train)->set_select({ind_fila,ind_col[7]}, X[ind*9+7]);
		(*x_train)->set_select({ind_fila,ind_col[8]}, X[ind*9+8]);
		for (j=0; j<num_clases; j++) {
			(*y_train)->set_select({ind_fila,ind_col[j]}, Y[ind*num_clases+j]);
		}
	}
	for (i=num_train; i<num_train+num_val; i++) {
		ind = pos[i];
		sprintf(ind_fila, "%d", i-num_train);
		(*x_val)->set_select({ind_fila,ind_col[0]}, X[ind*9]);
		(*x_val)->set_select({ind_fila,ind_col[1]}, X[ind*9+1]);
		(*x_val)->set_select({ind_fila,ind_col[2]}, X[ind*9+2]);
		(*x_val)->set_select({ind_fila,ind_col[3]}, X[ind*9+3]);
		(*x_val)->set_select({ind_fila,ind_col[4]}, X[ind*9+4]);
		(*x_val)->set_select({ind_fila,ind_col[5]}, X[ind*9+5]);
		(*x_val)->set_select({ind_fila,ind_col[6]}, X[ind*9+6]);
		(*x_val)->set_select({ind_fila,ind_col[7]}, X[ind*9+7]);
		(*x_val)->set_select({ind_fila,ind_col[8]}, X[ind*9+8]);
		for (j=0; j<num_clases; j++) {
			(*y_val)->set_select({ind_fila,ind_col[j]}, Y[ind*num_clases+j]);
		}
	}
	for (i=num_train+num_val; i<num_train+num_val+num_test; i++) {
		ind = pos[i];
		sprintf(ind_fila, "%d", i-num_train-num_val);
		(*x_test)->set_select({ind_fila,ind_col[0]}, X[ind*9]);
		(*x_test)->set_select({ind_fila,ind_col[1]}, X[ind*9+1]);
		(*x_test)->set_select({ind_fila,ind_col[2]}, X[ind*9+2]);
		(*x_test)->set_select({ind_fila,ind_col[3]}, X[ind*9+3]);
		(*x_test)->set_select({ind_fila,ind_col[4]}, X[ind*9+4]);
		(*x_test)->set_select({ind_fila,ind_col[5]}, X[ind*9+5]);
		(*x_test)->set_select({ind_fila,ind_col[6]}, X[ind*9+6]);
		(*x_test)->set_select({ind_fila,ind_col[7]}, X[ind*9+7]);
		(*x_test)->set_select({ind_fila,ind_col[8]}, X[ind*9+8]);
		for (j=0; j<num_clases; j++) {
			(*y_test)->set_select({ind_fila,ind_col[j]}, Y[ind*num_clases+j]);
		}
	}
	free(X);
	free(Y);
}

float entrenamiento(model net, Tensor *x_train, Tensor *y_train, Tensor *x_val, Tensor *y_val, int num_clases,
			int max_epochs, int batch_size, int patience_es, int patience_lr, float lr_final, int *epochs,
			int id_hebra)
{
	vector<Tensor *> preds;
	Tensor *classes_y, *classes_pred;
    std::pair<unsigned int *, int> res;
	Tensor *dif;
    bool terminar = false;
	bool lr_reducida = false;
	float loss, mejor_loss;
    int iter_max = 0;
	int num_errores;
	int i, j;
    vector<vtensor> pesos;
    Tensor *xbatch = new Tensor({batch_size,9});
    Tensor *ybatch = new Tensor({batch_size,num_clases});
	int num_train = x_train->shape[0];
    int num_batches = num_train/batch_size;
	int num_val = x_val->shape[0];

#if (LOSS_ES == 1)
	mejor_loss = 1e20;
    i = 0;
    while ((! terminar) && (i < max_epochs)) {
//        fit(net, {x_train}, {y_train}, batch_size, 1);
        // Inicio fit
        reset_loss(net);
        for(j=0; j<num_batches; j++)  {
            next_batch({x_train,y_train}, {xbatch,ybatch});
            train_batch(net, {xbatch}, {ybatch});
        }
        // Fin fit

        preds = predict(net, {x_val});
        loss = tensorNN::categorical_cross_entropy(y_val, preds[0]);
        loss /= num_val;
        if (loss < mejor_loss) {
            mejor_loss = loss;
            pesos = get_parameters(net, true);
            iter_max = i;
        }
        else if ((i-iter_max >= patience_lr) && (! lr_reducida)) {
			setlr(net, {lr_final});
			lr_reducida = true;
		}
        else if (i-iter_max >= patience_es) {
            terminar = true;
        }
        i++;
        fprintf(stdout, "  Process %d, epoch %2d, best loss: %e, patience: %d epochs\n", id_hebra, i, mejor_loss, i-iter_max);
    }
#else
    classes_y = y_val->argmax({1}, false);
	mejor_loss = -1.0;
    i = 0;
    while ((! terminar) && (i < max_epochs)) {
//        fit(net, {x_train}, {y_train}, batch_size, 1);
        // Inicio fit
        reset_loss(net);
        for(j=0; j<num_batches; j++)  {
            next_batch({x_train,y_train}, {xbatch,ybatch});
            train_batch(net, {xbatch}, {ybatch});
        }
        // Fin fit

        preds = predict(net, {x_val});
        classes_pred = preds[0]->argmax({1}, false);
        dif = classes_y->sub(classes_pred);
        dif->abs_();
        res = dif->_nonzero();
        num_errores = res.second;
        loss = (dif->shape[0]-num_errores)*100.0 / (dif->shape[0]);
        if (loss > mejor_loss) {
            mejor_loss = loss;
            pesos = get_parameters(net, true);
            iter_max = i;
        }
        else if ((i-iter_max >= patience_lr) && (! lr_reducida)) {
			setlr(net, {lr_final});
			lr_reducida = true;
		}
        else if (i-iter_max >= patience_es) {
            terminar = true;
        }
        i++;
        fprintf(stdout, "  Process %d, epoch %2d, best accuracy: %.4f, patience: %d epochs\n", id_hebra, i, mejor_loss, i-iter_max);
    }
#endif

	// Asignamos el número de epochs realizados
	*epochs = i;
	// Asignamos los mejores pesos obtenidos
	set_parameters(net, pesos);

	delete xbatch;
	delete ybatch;

	return mejor_loss;
}

void evaluar_modelo(model net, Tensor *x_train, Tensor *y_train, Tensor *x_val, Tensor *y_val, Tensor *x_test,
			Tensor *y_test, int num_clases)
{
	vector<Tensor *> preds;
	Tensor *classes_y, *classes_pred;
    std::pair<unsigned int *, int> res;
	Tensor *dif;
	float loss;
	int num_errores;

    // Evaluate train
    fprintf(stdout, "Training\n");
    classes_y = y_train->argmax({1}, false);
    preds = predict(net, {x_train});
    classes_pred = preds[0]->argmax({1}, false);
    dif = classes_y->sub(classes_pred);
    dif->abs_();
    res = dif->_nonzero();
    num_errores = res.second;
    loss = (dif->shape[0]-num_errores)*100.0 / (dif->shape[0]);
    fprintf(stdout, "  accuracy: %d/%d (%.2f %%)\n", dif->shape[0]-num_errores, dif->shape[0], loss);

    // Evaluate val
    fprintf(stdout, "Validation\n");
    classes_y = y_val->argmax({1}, false);
    preds = predict(net, {x_val});
    classes_pred = preds[0]->argmax({1}, false);
    dif = classes_y->sub(classes_pred);
    dif->abs_();
    res = dif->_nonzero();
    num_errores = res.second;
    loss = (dif->shape[0]-num_errores)*100.0 / (dif->shape[0]);
    fprintf(stdout, "  accuracy: %d/%d (%.2f %%)\n", dif->shape[0]-num_errores, dif->shape[0], loss);

    // Evaluate test
    fprintf(stdout, "Test\n");
    classes_y = y_test->argmax({1}, false);
    preds = predict(net, {x_test});
    classes_pred = preds[0]->argmax({1}, false);
    dif = classes_y->sub(classes_pred);
    dif->abs_();
    res = dif->_nonzero();
    num_errores = res.second;
    loss = (dif->shape[0]-num_errores)*100.0 / (dif->shape[0]);
    fprintf(stdout, "  accuracy: %d/%d (%.2f %%)\n", dif->shape[0]-num_errores, dif->shape[0], loss);

    fprintf(stdout, "Test\n");
    matrizDeConfusion(num_clases, classes_y, classes_pred);
}

