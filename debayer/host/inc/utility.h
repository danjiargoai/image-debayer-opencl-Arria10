// 
unsigned char* get_binary(const char * name, size_t* length)
{
	FILE *fp = fopen(name, "rb");
	printf("%s\n", name);
	assert (fp != NULL);
	fseek (fp, 0, SEEK_END);
	*length = ftell (fp);
	unsigned char *binaries = (unsigned char*)malloc(sizeof(unsigned char) **length);
	rewind (fp);
	fread (binaries, *length, 1, fp);
	fclose (fp);
	return binaries;
}
