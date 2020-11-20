#include<stdio.h>
#include<stdlib.h>

typedef struct dict_element dict_element;
struct dict_element{
    int value;
    char* key;
};

struct list{
     dict_element*  item;
     struct list* head;
     struct list* next;
};

typedef struct hash_map hash_map;
struct hash_map{
    struct list* itemList;
    int count;
    struct list* head;
};

char* compare(char* a,char* b){
    int i=0;
    while(a[i]!="\0" && b[i]!="\0")
    {
        if(a[i]>b[i])
            return b;
        else if(a[i]<b[i])
            return a;
    }
    return "same";
}


struct list* createList(){
    
}

struct list* deleteNode(){

}


struct list* insertNode(dict_element* element,hash_map* map){
    char* key = element->key;
    int pos = findPosition(key,map->itemList);

    dict_element* traverse = map->itemList->head;
    while(pos--)
    {
        traverse = traverse->next;
    }

    traverse->next = element;
    element->next = 
}

struct dict_element* find(){

}


// int main(){
//     node start = (node)malloc(sizeof(struct dict_element));
//     start->key = (char*)malloc(1000*sizeof(char));
//     start->value = 100;
//     start->key = "mohan";


//     start->key = "sohan";
// }