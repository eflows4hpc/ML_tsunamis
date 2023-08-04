#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "eddl.cxx"
#include "eddl/serialization/onnx/eddl_onnx.h"

void mostrarFormatoPrograma(char *argv[])
{
	cerr << "Use:" << endl;
	cerr << argv[0] << " modelFile DataToInferFile inferParametersFile" << endl;
}

int main(int argc, char **argv)
{
	int i, j;
	int err = 0;
	float val_min, factor;
	bool use_cpu = true;
	string directorio, fich_modelo;
	string fich_datosX, fich_par;
	string fich_datosY("predicted.txt");
	int num_ejemplos;
	int num_puntos = 6;
	int id_gpu = 0;

	Tensor *x_test, *y_test;
	model net;

	if (argc < 4) {
		mostrarFormatoPrograma(argv);
		return EXIT_SUCCESS;
	}

	// Fichero de datos
	fich_modelo = argv[1];
	fich_datosX = argv[2];
	fich_par = argv[3];
	if (! existeFichero(fich_modelo)) {
		cerr << "Error: File '" << fich_modelo << "' not found" << endl;
		err = 1;
	}
	else if (! existeFichero(fich_datosX)) {
		cerr << "Error: File '" << fich_datosX << "' not found" << endl;
		err = 1;
	}
	else if (! existeFichero(fich_par)) {
		cerr << "Error: File '" << fich_par << "' not found" << endl;
		err = 1;
	}
	if (err != 0) {
		return EXIT_FAILURE;
	}

    // Cargamos el modelo
	load_model(&net, fich_modelo, use_cpu, id_gpu);
	// Cargamos los datos a inferir y los ponemos en x_test
	cargarDatosAInferir(fich_datosX, &num_ejemplos, &x_test);
	// Cargamos los parámetros de inferencia, necesarios para denormalizar las salidas
	cargarParametrosInferencia(fich_par, &val_min, &factor);

    fprintf(stdout, "Predicting\n");
	prededir_max_height(net, &x_test, &y_test, num_puntos, val_min, factor);
    fprintf(stdout, "Saving %s\n", fich_datosY.c_str());
	guardarTensor(fich_datosY, y_test, num_ejemplos, num_puntos);

	delete x_test;
	delete y_test;
	delete net;

	return EXIT_SUCCESS;
}

