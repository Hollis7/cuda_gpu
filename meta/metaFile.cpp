#include <iostream>
#include <string>
#include <cstdlib>

int main() {
    char **p = NULL;
    int i;
    p = (char **)calloc(10, sizeof(char *));
    printf("%s",p[0]);
    for (i = 0; i < 10; i++) {
        p[i] = (char *)malloc(sizeof(char) * 2);
        // assign the character '0' plus the current index to the first element of the string
        (*p[i]) = '0' + i;
        // add a null terminator character at the end of the string
        *(p[i] + 1) = '\0';
        printf("%s\n", p[i]);
    }
    for (i = 0; i < 10; i++) {
        free(p[i]);
    }
    free(p);
    return 0;
}
