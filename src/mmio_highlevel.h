#include "mmio.h"
void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric,
                  index_t **csrRowPtr, int **csrColIdx, value_t **csrVal,
                  char *filename)
{
    int m_tmp, n_tmp;
    unsigned long long nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    index_t nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    index_t *csrRowPtr_counter = (index_t *)malloc((m_tmp + 1) * sizeof(index_t));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(index_t));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    value_t *csrVal_tmp = (value_t *)malloc(nnz_mtx_report * sizeof(value_t));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (index_t i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }
    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (index_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    exclusive_scan(csrRowPtr_counter, m_tmp + 1);

    index_t *csrRowPtr_alias = (index_t *)malloc((m_tmp + 1) * sizeof(index_t));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    value_t *csrVal_alias = (value_t *)malloc(nnz_tmp * sizeof(value_t));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(index_t));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(index_t));

    if (isSymmetric_tmp)
    {
        for (index_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                index_t offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                index_t offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (index_t i = 0; i < nnz_mtx_report; i++)
        {
            index_t offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
void csr2csc(const int m,
             const int n,
             const index_t nnz,
             const index_t *csrRowPtr,
             const index_t *csrColIdx,
             const value_t *csrVal,
             index_t *cscRowIdx,
             index_t *cscColPtr,
             value_t *cscVal)
{
    // histogram in column pointer
    memset(cscColPtr, 0, sizeof(index_t) * (n + 1));
    for (index_t i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    index_t *cscColIncr = (index_t *)malloc(sizeof(index_t) * (n + 1));
    memcpy(cscColIncr, cscColPtr, sizeof(index_t) * (n + 1));
    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (index_t j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }
    free(cscColIncr);
}
