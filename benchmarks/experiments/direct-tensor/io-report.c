__attribute__((destructor)) static void report(void) {
    if (!atomic_load(&calls)) return;
    char name[4096];
    snprintf(name,sizeof(name),"%s/%d.io",getenv("H3_TRIAL_DIR"),getpid());
    FILE *f=fopen(name,"w");
    if(f){fprintf(f,"%llu %llu %llu %llu\n",atomic_load(&calls),atomic_load(&logical_bytes),atomic_load(&aligned_bytes),atomic_load(&errors));fclose(f);}
}
