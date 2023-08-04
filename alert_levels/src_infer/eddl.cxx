#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <sys/stat.h> 
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

using namespace std;
using namespace eddl;

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

void cargarDatosAInferir(string nombre_fichX, int *num_ejemplos, Tensor **x_test)
{
	ifstream fichX(nombre_fichX);
	float val, val_max;
	int i, j, ind;
	char ind_fila[16];
	const char *ind_col[9] = {"0","1","2","3","4","5","6","7","8"};
	int inc_reserva = 10000;
	int num_elems_reserva;
	float *X = (float *) malloc(inc_reserva*9*sizeof(float));
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
	fichX.close();

	*x_test = new Tensor({*num_ejemplos,9});

	for (i=0; i<(*num_ejemplos); i++) {
		sprintf(ind_fila, "%d", i);
		(*x_test)->set_select({ind_fila,ind_col[0]}, X[i*9]);
		(*x_test)->set_select({ind_fila,ind_col[1]}, X[i*9+1]);
		(*x_test)->set_select({ind_fila,ind_col[2]}, X[i*9+2]);
		(*x_test)->set_select({ind_fila,ind_col[3]}, X[i*9+3]);
		(*x_test)->set_select({ind_fila,ind_col[4]}, X[i*9+4]);
		(*x_test)->set_select({ind_fila,ind_col[5]}, X[i*9+5]);
		(*x_test)->set_select({ind_fila,ind_col[6]}, X[i*9+6]);
		(*x_test)->set_select({ind_fila,ind_col[7]}, X[i*9+7]);
		(*x_test)->set_select({ind_fila,ind_col[8]}, X[i*9+8]);
	}
	free(X);
}

/*****************/
/* Deep learning */
/*****************/

void load_model(model *net, string nombre_fich, bool use_cpu, int id_gpu)
{
	compserv cs = nullptr;
	vector<int> gpus(GPUS_PER_NODE, 0);

	*net = import_net_from_onnx_file(nombre_fich);
    (*net)->verbosity_level = 0;

	// dot de graphviz debe estar instalado
//	plot(*net, "model.pdf");

	if (use_cpu) {
		cs = CS_CPU(1);
	} else {
		// Se utiliza la gpu id_gpu
		gpus[id_gpu] = 1;
		cs = CS_GPU(gpus);
	}

	build(*net, adam(1e-3), {"categorical_cross_entropy"}, {"categorical_accuracy"}, cs, false);

//	summary(*net);
}

void prededir_niveles_alerta(model net, Tensor **x_test, Tensor **y_test)
{
	vector<Tensor *> preds;

	preds = predict(net, {*x_test});
	*y_test = preds[0];
}

/****************************/
/* Guardado de predicciones */
/****************************/

void guardarTensor(string nombre_fich, Tensor *t, int num_ejemplos)
{
	FILE *fp;
	Tensor *classes;
	int i, j;
	int val;

	classes = t->argmax({1}, false);
	fp = fopen((char *) nombre_fich.c_str(), "wt");
	for (i=0; i<num_ejemplos; i++) {
		val = classes->ptr[i];
		fprintf(fp, "%d\n", val);
	}
	fclose(fp);
}

