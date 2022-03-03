template <typename T>
void mass_apply(int Ne, const T* xe, const T* phi, const T* detJ, T* ye, int ndofs,
                int block_size);