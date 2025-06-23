#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    FILE *arq;
    int nLinhas, nColunas;
    float celula;

    if((arq = fopen(argv[1], "w")) == NULL) {
        fprintf(stderr, "Impossível abrir o arquivo %s para escrita.\n", argv[1]);
        exit(-1);
    }

    nLinhas = atoi(argv[2]);
    nColunas = atoi(argv[3]);

    for(int linha=0; linha<nLinhas; linha++) {
        for(int coluna=0; coluna<nColunas; coluna++) {
            celula = (linha * nLinhas + coluna)/0.01;
            fwrite(&celula, sizeof(float), 1, arq);
        }
    }

    fclose(arq);
    return 0;
}