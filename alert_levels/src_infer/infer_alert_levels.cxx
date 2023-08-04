#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "eddl.cxx"
#include "eddl/serialization/onnx/eddl_onnx.h"

void mostrarFormatoPrograma(char *argv[])
{
	cerr << "Use:" << endl;
	cerr << argv[0] << " modelFile DataToInferFile" << endl;
}

int main(int argc, char **argv)
{
	int i, j;
	int err = 0;
	bool use_cpu = true;
	string directorio, fich_modelo;
	string fich_datosX;
	string fich_datosY("predicted.txt");
	int num_ejemplos;
	int id_gpu = 0;

	Tensor *x_test, *y_test;
	model net;

	if (argc < 3) {
		mostrarFormatoPrograma(argv);
		return EXIT_SUCCESS;
	}

	// Fichero de datos
	fich_modelo = argv[1];
	fich_datosX = argv[2];
	if (! existeFichero(fich_modelo)) {
		cerr << "Error: File '" << fich_modelo << "' not found" << endl;
		err = 1;
	}
	else if (! existeFichero(fich_datosX)) {
		cerr << "Error: File '" << fich_datosX << "' not found" << endl;
		err = 1;
	}
	if (err != 0) {
		return EXIT_FAILURE;
	}

    // Cargamos el modelo
	load_model(&net, fich_modelo, use_cpu, id_gpu);
	// Cargamos los datos a inferir y los ponemos en x_test
	cargarDatosAInferir(fich_datosX, &num_ejemplos, &x_test);

    fprintf(stdout, "Predicting\n");
	prededir_niveles_alerta(net, &x_test, &y_test);
    fprintf(stdout, "Saving %s\n", fich_datosY.c_str());
	guardarTensor(fich_datosY, y_test, num_ejemplos);

	delete x_test;
	delete y_test;
	delete net;

	return EXIT_SUCCESS;
}

