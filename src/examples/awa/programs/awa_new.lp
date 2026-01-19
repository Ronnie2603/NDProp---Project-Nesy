% -------------------------
% Animals (50)
% -------------------------
animal(antelope).
animal(grizzly_bear).
animal(killer_whale).
animal(beaver).
animal(dalmatian).
animal(persian_cat).
animal(horse).
animal(german_shepherd).
animal(blue_whale).
animal(siamese_cat).
animal(skunk).
animal(mole).
animal(tiger).
animal(hippopotamus).
animal(leopard).
animal(moose).
animal(spider_monkey).
animal(humpback_whale).
animal(elephant).
animal(gorilla).
animal(ox).
animal(fox).
animal(sheep).
animal(seal).
animal(chimpanzee).
animal(hamster).
animal(squirrel).
animal(rhinoceros).
animal(rabbit).
animal(bat).
animal(giraffe).
animal(wolf).
animal(chihuahua).
animal(rat).
animal(weasel).
animal(otter).
animal(buffalo).
animal(zebra).
animal(giant_panda).
animal(deer).
animal(bobcat).
animal(pig).
animal(lion).
animal(mouse).
animal(polar_bear).
animal(collie).
animal(walrus).
animal(raccoon).
animal(cow).
animal(dolphin).

% -------------------------
% "Choice" (normal rules)
% (by default negation)
% -------------------------
sel(A)  :- animal(A), not nsel(A).
nsel(A) :- animal(A), not sel(A).

% -------------------------
% EXACTLY ONE sel/1
% (NO CONSTRAINTS!)
% -------------------------

% at least one selected:
some_sel :- sel(A).
bad_none :- not some_sel, not bad_none.

% at most one selected:
two_sel :- sel(A1), sel(A2), A1 != A2.
bad_two :- two_sel, not bad_two.

% -------------------------
% Attributes (your big attr/2 block goes here)
% -------------------------
% attr(antelope,furry).
% ...
% (keep all your attr/2 facts exactly as-is)


% --- Attributes ---
attr(antelope,furry).
attr(antelope,toughskin).
attr(antelope,big).
attr(antelope,lean).
attr(antelope,hooves).
attr(antelope,longleg).
attr(antelope,tail).
attr(antelope,chewteeth).
attr(antelope,horns).
attr(antelope,walks).
attr(antelope,fast).
attr(antelope,strong).
attr(antelope,muscle).
attr(antelope,quadrapedal).
attr(antelope,active).
attr(antelope,agility).
attr(antelope,vegetation).
attr(antelope,forager).
attr(antelope,grazer).
attr(antelope,newworld).
attr(antelope,oldworld).
attr(antelope,plains).
attr(antelope,fields).
attr(antelope,mountains).
attr(antelope,ground).
attr(antelope,timid).
attr(antelope,group).

attr(grizzly_bear,black).
attr(grizzly_bear,brown).
attr(grizzly_bear,furry).
attr(grizzly_bear,toughskin).
attr(grizzly_bear,big).
attr(grizzly_bear,bulbous).
attr(grizzly_bear,paws).
attr(grizzly_bear,chewteeth).
attr(grizzly_bear,meatteeth).
attr(grizzly_bear,claws).
attr(grizzly_bear,walks).
attr(grizzly_bear,fast).
attr(grizzly_bear,slow).
attr(grizzly_bear,strong).
attr(grizzly_bear,muscle).
attr(grizzly_bear,bipedal).
attr(grizzly_bear,quadrapedal).
attr(grizzly_bear,active).
attr(grizzly_bear,inactive).
attr(grizzly_bear,nocturnal).
attr(grizzly_bear,hibernate).
attr(grizzly_bear,fish).
attr(grizzly_bear,meat).
attr(grizzly_bear,forager).
attr(grizzly_bear,hunter).
attr(grizzly_bear,stalker).
attr(grizzly_bear,newworld).
attr(grizzly_bear,forest).
attr(grizzly_bear,mountains).
attr(grizzly_bear,ground).
attr(grizzly_bear,cave).
attr(grizzly_bear,fierce).
attr(grizzly_bear,smart).
attr(grizzly_bear,solitary).

attr(killer_whale,black).
attr(killer_whale,white).
attr(killer_whale,patches).
attr(killer_whale,spots).
attr(killer_whale,hairless).
attr(killer_whale,toughskin).
attr(killer_whale,big).
attr(killer_whale,bulbous).
attr(killer_whale,lean).
attr(killer_whale,flippers).
attr(killer_whale,tail).
attr(killer_whale,meatteeth).
attr(killer_whale,strainteeth).
attr(killer_whale,swims).
attr(killer_whale,fast).
attr(killer_whale,strong).
attr(killer_whale,active).
attr(killer_whale,agility).
attr(killer_whale,fish).
attr(killer_whale,meat).
attr(killer_whale,plankton).
attr(killer_whale,hunter).
attr(killer_whale,arctic).
attr(killer_whale,coastal).
attr(killer_whale,ocean).
attr(killer_whale,water).
attr(killer_whale,fierce).
attr(killer_whale,smart).
attr(killer_whale,group).

attr(beaver,brown).
attr(beaver,furry).
attr(beaver,toughskin).
attr(beaver,small).
attr(beaver,bulbous).
attr(beaver,pads).
attr(beaver,paws).
attr(beaver,tail).
attr(beaver,chewteeth).
attr(beaver,buckteeth).
attr(beaver,strainteeth).
attr(beaver,claws).
attr(beaver,swims).
attr(beaver,fast).
attr(beaver,strong).
attr(beaver,muscle).
attr(beaver,quadrapedal).
attr(beaver,active).
attr(beaver,nocturnal).
attr(beaver,hibernate).
attr(beaver,agility).
attr(beaver,fish).
attr(beaver,newworld).
attr(beaver,coastal).
attr(beaver,ground).
attr(beaver,water).
attr(beaver,timid).
attr(beaver,smart).
attr(beaver,group).
attr(beaver,solitary).
attr(beaver,nestspot).

attr(dalmatian,black).
attr(dalmatian,white).
attr(dalmatian,patches).
attr(dalmatian,spots).
attr(dalmatian,furry).
attr(dalmatian,hairless).
attr(dalmatian,big).
attr(dalmatian,lean).
attr(dalmatian,paws).
attr(dalmatian,longleg).
attr(dalmatian,tail).
attr(dalmatian,chewteeth).
attr(dalmatian,meatteeth).
attr(dalmatian,walks).
attr(dalmatian,fast).
attr(dalmatian,strong).
attr(dalmatian,muscle).
attr(dalmatian,quadrapedal).
attr(dalmatian,active).
attr(dalmatian,agility).
attr(dalmatian,meat).
attr(dalmatian,newworld).
attr(dalmatian,oldworld).
attr(dalmatian,ground).
attr(dalmatian,timid).
attr(dalmatian,smart).
attr(dalmatian,group).
attr(dalmatian,solitary).
attr(dalmatian,domestic).

attr(persian_cat,white).
attr(persian_cat,blue).
attr(persian_cat,gray).
attr(persian_cat,furry).
attr(persian_cat,small).
attr(persian_cat,bulbous).
attr(persian_cat,pads).
attr(persian_cat,paws).
attr(persian_cat,tail).
attr(persian_cat,chewteeth).
attr(persian_cat,meatteeth).
attr(persian_cat,claws).
attr(persian_cat,walks).
attr(persian_cat,fast).
attr(persian_cat,slow).
attr(persian_cat,weak).
attr(persian_cat,quadrapedal).
attr(persian_cat,inactive).
attr(persian_cat,agility).
attr(persian_cat,fish).
attr(persian_cat,meat).
attr(persian_cat,newworld).
attr(persian_cat,oldworld).
attr(persian_cat,ground).
attr(persian_cat,timid).
attr(persian_cat,smart).
attr(persian_cat,solitary).
attr(persian_cat,domestic).

attr(horse,black).
attr(horse,white).
attr(horse,brown).
attr(horse,gray).
attr(horse,patches).
attr(horse,furry).
attr(horse,toughskin).
attr(horse,big).
attr(horse,lean).
attr(horse,hooves).
attr(horse,longleg).
attr(horse,longneck).
attr(horse,tail).
attr(horse,chewteeth).
attr(horse,buckteeth).
attr(horse,smelly).
attr(horse,walks).
attr(horse,fast).
attr(horse,strong).
attr(horse,muscle).
attr(horse,quadrapedal).
attr(horse,active).
attr(horse,agility).
attr(horse,vegetation).
attr(horse,grazer).
attr(horse,newworld).
attr(horse,oldworld).
attr(horse,plains).
attr(horse,fields).
attr(horse,ground).
attr(horse,timid).
attr(horse,smart).
attr(horse,group).
attr(horse,domestic).

attr(german_shepherd,black).
attr(german_shepherd,brown).
attr(german_shepherd,gray).
attr(german_shepherd,patches).
attr(german_shepherd,furry).
attr(german_shepherd,big).
attr(german_shepherd,lean).
attr(german_shepherd,pads).
attr(german_shepherd,paws).
attr(german_shepherd,longleg).
attr(german_shepherd,tail).
attr(german_shepherd,chewteeth).
attr(german_shepherd,meatteeth).
attr(german_shepherd,claws).
attr(german_shepherd,smelly).
attr(german_shepherd,walks).
attr(german_shepherd,fast).
attr(german_shepherd,strong).
attr(german_shepherd,muscle).
attr(german_shepherd,quadrapedal).
attr(german_shepherd,active).
attr(german_shepherd,agility).
attr(german_shepherd,meat).
attr(german_shepherd,hunter).
attr(german_shepherd,stalker).
attr(german_shepherd,newworld).
attr(german_shepherd,oldworld).
attr(german_shepherd,plains).
attr(german_shepherd,ground).
attr(german_shepherd,fierce).
attr(german_shepherd,smart).
attr(german_shepherd,solitary).
attr(german_shepherd,domestic).

attr(blue_whale,blue).
attr(blue_whale,gray).
attr(blue_whale,spots).
attr(blue_whale,hairless).
attr(blue_whale,toughskin).
attr(blue_whale,big).
attr(blue_whale,bulbous).
attr(blue_whale,flippers).
attr(blue_whale,tail).
attr(blue_whale,strainteeth).
attr(blue_whale,swims).
attr(blue_whale,fast).
attr(blue_whale,slow).
attr(blue_whale,strong).
attr(blue_whale,muscle).
attr(blue_whale,inactive).
attr(blue_whale,fish).
attr(blue_whale,plankton).
attr(blue_whale,skimmer).
attr(blue_whale,newworld).
attr(blue_whale,oldworld).
attr(blue_whale,arctic).
attr(blue_whale,ocean).
attr(blue_whale,water).
attr(blue_whale,timid).
attr(blue_whale,smart).
attr(blue_whale,group).
attr(blue_whale,solitary).

attr(siamese_cat,black).
attr(siamese_cat,white).
attr(siamese_cat,brown).
attr(siamese_cat,gray).
attr(siamese_cat,patches).
attr(siamese_cat,furry).
attr(siamese_cat,small).
attr(siamese_cat,lean).
attr(siamese_cat,pads).
attr(siamese_cat,paws).
attr(siamese_cat,longleg).
attr(siamese_cat,tail).
attr(siamese_cat,chewteeth).
attr(siamese_cat,meatteeth).
attr(siamese_cat,claws).
attr(siamese_cat,walks).
attr(siamese_cat,fast).
attr(siamese_cat,weak).
attr(siamese_cat,muscle).
attr(siamese_cat,quadrapedal).
attr(siamese_cat,active).
attr(siamese_cat,inactive).
attr(siamese_cat,nocturnal).
attr(siamese_cat,agility).
attr(siamese_cat,fish).
attr(siamese_cat,meat).
attr(siamese_cat,hunter).
attr(siamese_cat,stalker).
attr(siamese_cat,newworld).
attr(siamese_cat,oldworld).
attr(siamese_cat,ground).
attr(siamese_cat,fierce).
attr(siamese_cat,timid).
attr(siamese_cat,smart).
attr(siamese_cat,solitary).
attr(siamese_cat,domestic).

attr(skunk,black).
attr(skunk,white).
attr(skunk,stripes).
attr(skunk,furry).
attr(skunk,small).
attr(skunk,bulbous).
attr(skunk,pads).
attr(skunk,paws).
attr(skunk,tail).
attr(skunk,smelly).
attr(skunk,walks).
attr(skunk,fast).
attr(skunk,slow).
attr(skunk,weak).
attr(skunk,quadrapedal).
attr(skunk,active).
attr(skunk,nocturnal).
attr(skunk,hibernate).
attr(skunk,vegetation).
attr(skunk,forager).
attr(skunk,newworld).
attr(skunk,oldworld).
attr(skunk,forest).
attr(skunk,fields).
attr(skunk,ground).
attr(skunk,timid).
attr(skunk,solitary).

attr(mole,black).
attr(mole,brown).
attr(mole,gray).
attr(mole,furry).
attr(mole,small).
attr(mole,bulbous).
attr(mole,lean).
attr(mole,paws).
attr(mole,chewteeth).
attr(mole,buckteeth).
attr(mole,claws).
attr(mole,tunnels).
attr(mole,walks).
attr(mole,fast).
attr(mole,slow).
attr(mole,weak).
attr(mole,quadrapedal).
attr(mole,active).
attr(mole,nocturnal).
attr(mole,hibernate).
attr(mole,agility).
attr(mole,vegetation).
attr(mole,insects).
attr(mole,forager).
attr(mole,newworld).
attr(mole,oldworld).
attr(mole,plains).
attr(mole,forest).
attr(mole,fields).
attr(mole,ground).
attr(mole,timid).
attr(mole,solitary).
attr(mole,nestspot).

attr(tiger,black).
attr(tiger,white).
attr(tiger,orange).
attr(tiger,stripes).
attr(tiger,furry).
attr(tiger,big).
attr(tiger,lean).
attr(tiger,pads).
attr(tiger,paws).
attr(tiger,tail).
attr(tiger,chewteeth).
attr(tiger,meatteeth).
attr(tiger,buckteeth).
attr(tiger,claws).
attr(tiger,smelly).
attr(tiger,walks).
attr(tiger,fast).
attr(tiger,strong).
attr(tiger,muscle).
attr(tiger,quadrapedal).
attr(tiger,active).
attr(tiger,nocturnal).
attr(tiger,agility).
attr(tiger,meat).
attr(tiger,hunter).
attr(tiger,stalker).
attr(tiger,oldworld).
attr(tiger,bush).
attr(tiger,plains).
attr(tiger,forest).
attr(tiger,jungle).
attr(tiger,ground).
attr(tiger,fierce).
attr(tiger,smart).
attr(tiger,group).
attr(tiger,solitary).
attr(tiger,nestspot).

attr(hippopotamus,gray).
attr(hippopotamus,hairless).
attr(hippopotamus,toughskin).
attr(hippopotamus,big).
attr(hippopotamus,bulbous).
attr(hippopotamus,chewteeth).
attr(hippopotamus,strainteeth).
attr(hippopotamus,swims).
attr(hippopotamus,walks).
attr(hippopotamus,slow).
attr(hippopotamus,strong).
attr(hippopotamus,muscle).
attr(hippopotamus,quadrapedal).
attr(hippopotamus,inactive).
attr(hippopotamus,fish).
attr(hippopotamus,vegetation).
attr(hippopotamus,oldworld).
attr(hippopotamus,jungle).
attr(hippopotamus,ground).
attr(hippopotamus,water).
attr(hippopotamus,timid).
attr(hippopotamus,solitary).

attr(leopard,black).
attr(leopard,brown).
attr(leopard,yellow).
attr(leopard,patches).
attr(leopard,spots).
attr(leopard,furry).
attr(leopard,big).
attr(leopard,lean).
attr(leopard,paws).
attr(leopard,longleg).
attr(leopard,tail).
attr(leopard,meatteeth).
attr(leopard,claws).
attr(leopard,walks).
attr(leopard,fast).
attr(leopard,strong).
attr(leopard,muscle).
attr(leopard,quadrapedal).
attr(leopard,active).
attr(leopard,nocturnal).
attr(leopard,agility).
attr(leopard,fish).
attr(leopard,meat).
attr(leopard,forager).
attr(leopard,hunter).
attr(leopard,stalker).
attr(leopard,newworld).
attr(leopard,oldworld).
attr(leopard,bush).
attr(leopard,plains).
attr(leopard,jungle).
attr(leopard,mountains).
attr(leopard,ground).
attr(leopard,tree).
attr(leopard,fierce).
attr(leopard,smart).
attr(leopard,solitary).
attr(leopard,nestspot).

attr(moose,brown).
attr(moose,furry).
attr(moose,toughskin).
attr(moose,big).
attr(moose,bulbous).
attr(moose,hooves).
attr(moose,longleg).
attr(moose,longneck).
attr(moose,tail).
attr(moose,chewteeth).
attr(moose,horns).
attr(moose,smelly).
attr(moose,walks).
attr(moose,fast).
attr(moose,slow).
attr(moose,strong).
attr(moose,muscle).
attr(moose,quadrapedal).
attr(moose,inactive).
attr(moose,vegetation).
attr(moose,forager).
attr(moose,grazer).
attr(moose,newworld).
attr(moose,oldworld).
attr(moose,arctic).
attr(moose,plains).
attr(moose,forest).
attr(moose,fields).
attr(moose,mountains).
attr(moose,ground).
attr(moose,timid).
attr(moose,group).
attr(moose,solitary).

attr(spider_monkey,black).
attr(spider_monkey,brown).
attr(spider_monkey,gray).
attr(spider_monkey,furry).
attr(spider_monkey,small).
attr(spider_monkey,lean).
attr(spider_monkey,hands).
attr(spider_monkey,paws).
attr(spider_monkey,longleg).
attr(spider_monkey,tail).
attr(spider_monkey,chewteeth).
attr(spider_monkey,walks).
attr(spider_monkey,fast).
attr(spider_monkey,strong).
attr(spider_monkey,muscle).
attr(spider_monkey,bipedal).
attr(spider_monkey,quadrapedal).
attr(spider_monkey,active).
attr(spider_monkey,agility).
attr(spider_monkey,vegetation).
attr(spider_monkey,forager).
attr(spider_monkey,grazer).
attr(spider_monkey,newworld).
attr(spider_monkey,oldworld).
attr(spider_monkey,forest).
attr(spider_monkey,jungle).
attr(spider_monkey,tree).
attr(spider_monkey,timid).
attr(spider_monkey,smart).
attr(spider_monkey,group).
attr(spider_monkey,nestspot).

attr(humpback_whale,black).
attr(humpback_whale,blue).
attr(humpback_whale,gray).
attr(humpback_whale,hairless).
attr(humpback_whale,toughskin).
attr(humpback_whale,big).
attr(humpback_whale,bulbous).
attr(humpback_whale,flippers).
attr(humpback_whale,tail).
attr(humpback_whale,strainteeth).
attr(humpback_whale,swims).
attr(humpback_whale,fast).
attr(humpback_whale,slow).
attr(humpback_whale,strong).
attr(humpback_whale,muscle).
attr(humpback_whale,inactive).
attr(humpback_whale,fish).
attr(humpback_whale,plankton).
attr(humpback_whale,skimmer).
attr(humpback_whale,newworld).
attr(humpback_whale,oldworld).
attr(humpback_whale,arctic).
attr(humpback_whale,coastal).
attr(humpback_whale,ocean).
attr(humpback_whale,water).
attr(humpback_whale,timid).
attr(humpback_whale,smart).
attr(humpback_whale,group).

attr(elephant,gray).
attr(elephant,hairless).
attr(elephant,toughskin).
attr(elephant,big).
attr(elephant,bulbous).
attr(elephant,longleg).
attr(elephant,tail).
attr(elephant,chewteeth).
attr(elephant,tusks).
attr(elephant,smelly).
attr(elephant,walks).
attr(elephant,slow).
attr(elephant,strong).
attr(elephant,muscle).
attr(elephant,quadrapedal).
attr(elephant,inactive).
attr(elephant,vegetation).
attr(elephant,grazer).
attr(elephant,oldworld).
attr(elephant,bush).
attr(elephant,jungle).
attr(elephant,ground).
attr(elephant,timid).
attr(elephant,smart).
attr(elephant,group).

attr(gorilla,black).
attr(gorilla,brown).
attr(gorilla,furry).
attr(gorilla,toughskin).
attr(gorilla,big).
attr(gorilla,bulbous).
attr(gorilla,hands).
attr(gorilla,longleg).
attr(gorilla,chewteeth).
attr(gorilla,meatteeth).
attr(gorilla,smelly).
attr(gorilla,walks).
attr(gorilla,fast).
attr(gorilla,strong).
attr(gorilla,muscle).
attr(gorilla,bipedal).
attr(gorilla,quadrapedal).
attr(gorilla,active).
attr(gorilla,agility).
attr(gorilla,meat).
attr(gorilla,vegetation).
attr(gorilla,forager).
attr(gorilla,oldworld).
attr(gorilla,forest).
attr(gorilla,jungle).
attr(gorilla,ground).
attr(gorilla,tree).
attr(gorilla,fierce).
attr(gorilla,smart).
attr(gorilla,group).
attr(gorilla,nestspot).

attr(ox,black).
attr(ox,white).
attr(ox,brown).
attr(ox,gray).
attr(ox,furry).
attr(ox,hairless).
attr(ox,toughskin).
attr(ox,big).
attr(ox,bulbous).
attr(ox,hooves).
attr(ox,tail).
attr(ox,chewteeth).
attr(ox,horns).
attr(ox,smelly).
attr(ox,walks).
attr(ox,slow).
attr(ox,strong).
attr(ox,muscle).
attr(ox,quadrapedal).
attr(ox,inactive).
attr(ox,vegetation).
attr(ox,grazer).
attr(ox,newworld).
attr(ox,oldworld).
attr(ox,bush).
attr(ox,plains).
attr(ox,fields).
attr(ox,ground).
attr(ox,timid).
attr(ox,solitary).
attr(ox,domestic).

attr(fox,brown).
attr(fox,orange).
attr(fox,red).
attr(fox,furry).
attr(fox,small).
attr(fox,lean).
attr(fox,pads).
attr(fox,paws).
attr(fox,tail).
attr(fox,chewteeth).
attr(fox,meatteeth).
attr(fox,claws).
attr(fox,walks).
attr(fox,fast).
attr(fox,strong).
attr(fox,muscle).
attr(fox,quadrapedal).
attr(fox,active).
attr(fox,nocturnal).
attr(fox,hibernate).
attr(fox,agility).
attr(fox,fish).
attr(fox,meat).
attr(fox,forager).
attr(fox,hunter).
attr(fox,stalker).
attr(fox,newworld).
attr(fox,oldworld).
attr(fox,plains).
attr(fox,forest).
attr(fox,fields).
attr(fox,ground).
attr(fox,fierce).
attr(fox,smart).
attr(fox,solitary).
attr(fox,nestspot).

attr(sheep,black).
attr(sheep,white).
attr(sheep,furry).
attr(sheep,bulbous).
attr(sheep,hooves).
attr(sheep,chewteeth).
attr(sheep,smelly).
attr(sheep,walks).
attr(sheep,slow).
attr(sheep,weak).
attr(sheep,quadrapedal).
attr(sheep,inactive).
attr(sheep,vegetation).
attr(sheep,grazer).
attr(sheep,newworld).
attr(sheep,oldworld).
attr(sheep,plains).
attr(sheep,fields).
attr(sheep,mountains).
attr(sheep,ground).
attr(sheep,timid).
attr(sheep,group).
attr(sheep,domestic).

attr(seal,black).
attr(seal,white).
attr(seal,brown).
attr(seal,gray).
attr(seal,spots).
attr(seal,hairless).
attr(seal,toughskin).
attr(seal,big).
attr(seal,small).
attr(seal,flippers).
attr(seal,tail).
attr(seal,swims).
attr(seal,fast).
attr(seal,slow).
attr(seal,strong).
attr(seal,active).
attr(seal,inactive).
attr(seal,agility).
attr(seal,fish).
attr(seal,hunter).
attr(seal,newworld).
attr(seal,oldworld).
attr(seal,arctic).
attr(seal,coastal).
attr(seal,ocean).
attr(seal,water).
attr(seal,timid).
attr(seal,smart).
attr(seal,group).
attr(seal,domestic).

attr(chimpanzee,black).
attr(chimpanzee,brown).
attr(chimpanzee,furry).
attr(chimpanzee,toughskin).
attr(chimpanzee,big).
attr(chimpanzee,small).
attr(chimpanzee,lean).
attr(chimpanzee,hands).
attr(chimpanzee,longleg).
attr(chimpanzee,tail).
attr(chimpanzee,chewteeth).
attr(chimpanzee,meatteeth).
attr(chimpanzee,smelly).
attr(chimpanzee,walks).
attr(chimpanzee,fast).
attr(chimpanzee,strong).
attr(chimpanzee,muscle).
attr(chimpanzee,bipedal).
attr(chimpanzee,quadrapedal).
attr(chimpanzee,active).
attr(chimpanzee,agility).
attr(chimpanzee,vegetation).
attr(chimpanzee,insects).
attr(chimpanzee,forager).
attr(chimpanzee,newworld).
attr(chimpanzee,oldworld).
attr(chimpanzee,bush).
attr(chimpanzee,forest).
attr(chimpanzee,jungle).
attr(chimpanzee,mountains).
attr(chimpanzee,ground).
attr(chimpanzee,tree).
attr(chimpanzee,fierce).
attr(chimpanzee,timid).
attr(chimpanzee,smart).
attr(chimpanzee,group).
attr(chimpanzee,nestspot).
attr(chimpanzee,domestic).

attr(hamster,black).
attr(hamster,white).
attr(hamster,brown).
attr(hamster,gray).
attr(hamster,patches).
attr(hamster,furry).
attr(hamster,small).
attr(hamster,bulbous).
attr(hamster,paws).
attr(hamster,tail).
attr(hamster,chewteeth).
attr(hamster,buckteeth).
attr(hamster,claws).
attr(hamster,smelly).
attr(hamster,hops).
attr(hamster,tunnels).
attr(hamster,walks).
attr(hamster,fast).
attr(hamster,weak).
attr(hamster,quadrapedal).
attr(hamster,active).
attr(hamster,inactive).
attr(hamster,nocturnal).
attr(hamster,hibernate).
attr(hamster,agility).
attr(hamster,vegetation).
attr(hamster,forager).
attr(hamster,grazer).
attr(hamster,newworld).
attr(hamster,fields).
attr(hamster,ground).
attr(hamster,timid).
attr(hamster,solitary).
attr(hamster,nestspot).
attr(hamster,domestic).

attr(squirrel,brown).
attr(squirrel,gray).
attr(squirrel,furry).
attr(squirrel,small).
attr(squirrel,pads).
attr(squirrel,paws).
attr(squirrel,tail).
attr(squirrel,chewteeth).
attr(squirrel,buckteeth).
attr(squirrel,claws).
attr(squirrel,hops).
attr(squirrel,walks).
attr(squirrel,fast).
attr(squirrel,bipedal).
attr(squirrel,quadrapedal).
attr(squirrel,active).
attr(squirrel,hibernate).
attr(squirrel,agility).
attr(squirrel,vegetation).
attr(squirrel,forager).
attr(squirrel,newworld).
attr(squirrel,oldworld).
attr(squirrel,forest).
attr(squirrel,ground).
attr(squirrel,tree).
attr(squirrel,timid).
attr(squirrel,solitary).
attr(squirrel,nestspot).

attr(rhinoceros,gray).
attr(rhinoceros,hairless).
attr(rhinoceros,toughskin).
attr(rhinoceros,big).
attr(rhinoceros,bulbous).
attr(rhinoceros,hooves).
attr(rhinoceros,horns).
attr(rhinoceros,tusks).
attr(rhinoceros,smelly).
attr(rhinoceros,walks).
attr(rhinoceros,slow).
attr(rhinoceros,strong).
attr(rhinoceros,quadrapedal).
attr(rhinoceros,inactive).
attr(rhinoceros,vegetation).
attr(rhinoceros,grazer).
attr(rhinoceros,oldworld).
attr(rhinoceros,bush).
attr(rhinoceros,jungle).
attr(rhinoceros,ground).
attr(rhinoceros,fierce).
attr(rhinoceros,group).
attr(rhinoceros,solitary).

attr(rabbit,black).
attr(rabbit,white).
attr(rabbit,brown).
attr(rabbit,gray).
attr(rabbit,patches).
attr(rabbit,furry).
attr(rabbit,small).
attr(rabbit,bulbous).
attr(rabbit,paws).
attr(rabbit,tail).
attr(rabbit,chewteeth).
attr(rabbit,buckteeth).
attr(rabbit,hops).
attr(rabbit,fast).
attr(rabbit,weak).
attr(rabbit,quadrapedal).
attr(rabbit,active).
attr(rabbit,inactive).
attr(rabbit,agility).
attr(rabbit,vegetation).
attr(rabbit,forager).
attr(rabbit,grazer).
attr(rabbit,newworld).
attr(rabbit,oldworld).
attr(rabbit,bush).
attr(rabbit,plains).
attr(rabbit,forest).
attr(rabbit,fields).
attr(rabbit,ground).
attr(rabbit,timid).
attr(rabbit,group).
attr(rabbit,nestspot).
attr(rabbit,domestic).

attr(bat,black).
attr(bat,brown).
attr(bat,gray).
attr(bat,furry).
attr(bat,hairless).
attr(bat,toughskin).
attr(bat,small).
attr(bat,lean).
attr(bat,chewteeth).
attr(bat,meatteeth).
attr(bat,claws).
attr(bat,smelly).
attr(bat,flys).
attr(bat,fast).
attr(bat,weak).
attr(bat,muscle).
attr(bat,bipedal).
attr(bat,active).
attr(bat,inactive).
attr(bat,nocturnal).
attr(bat,hibernate).
attr(bat,agility).
attr(bat,meat).
attr(bat,vegetation).
attr(bat,insects).
attr(bat,forager).
attr(bat,hunter).
attr(bat,scavenger).
attr(bat,newworld).
attr(bat,oldworld).
attr(bat,forest).
attr(bat,jungle).
attr(bat,mountains).
attr(bat,tree).
attr(bat,cave).
attr(bat,fierce).
attr(bat,smart).
attr(bat,group).
attr(bat,nestspot).

attr(giraffe,brown).
attr(giraffe,orange).
attr(giraffe,yellow).
attr(giraffe,patches).
attr(giraffe,spots).
attr(giraffe,big).
attr(giraffe,lean).
attr(giraffe,hooves).
attr(giraffe,longleg).
attr(giraffe,longneck).
attr(giraffe,tail).
attr(giraffe,chewteeth).
attr(giraffe,horns).
attr(giraffe,smelly).
attr(giraffe,walks).
attr(giraffe,fast).
attr(giraffe,slow).
attr(giraffe,strong).
attr(giraffe,muscle).
attr(giraffe,quadrapedal).
attr(giraffe,active).
attr(giraffe,inactive).
attr(giraffe,vegetation).
attr(giraffe,grazer).
attr(giraffe,oldworld).
attr(giraffe,bush).
attr(giraffe,plains).
attr(giraffe,fields).
attr(giraffe,ground).
attr(giraffe,timid).
attr(giraffe,group).

attr(wolf,black).
attr(wolf,white).
attr(wolf,brown).
attr(wolf,gray).
attr(wolf,furry).
attr(wolf,big).
attr(wolf,lean).
attr(wolf,pads).
attr(wolf,paws).
attr(wolf,tail).
attr(wolf,chewteeth).
attr(wolf,meatteeth).
attr(wolf,claws).
attr(wolf,walks).
attr(wolf,fast).
attr(wolf,strong).
attr(wolf,muscle).
attr(wolf,quadrapedal).
attr(wolf,active).
attr(wolf,nocturnal).
attr(wolf,agility).
attr(wolf,meat).
attr(wolf,forager).
attr(wolf,hunter).
attr(wolf,scavenger).
attr(wolf,stalker).
attr(wolf,newworld).
attr(wolf,oldworld).
attr(wolf,arctic).
attr(wolf,plains).
attr(wolf,forest).
attr(wolf,mountains).
attr(wolf,ground).
attr(wolf,cave).
attr(wolf,fierce).
attr(wolf,smart).
attr(wolf,group).
attr(wolf,solitary).

attr(chihuahua,black).
attr(chihuahua,brown).
attr(chihuahua,gray).
attr(chihuahua,furry).
attr(chihuahua,small).
attr(chihuahua,lean).
attr(chihuahua,pads).
attr(chihuahua,paws).
attr(chihuahua,tail).
attr(chihuahua,meatteeth).
attr(chihuahua,claws).
attr(chihuahua,smelly).
attr(chihuahua,walks).
attr(chihuahua,fast).
attr(chihuahua,weak).
attr(chihuahua,quadrapedal).
attr(chihuahua,active).
attr(chihuahua,meat).
attr(chihuahua,newworld).
attr(chihuahua,oldworld).
attr(chihuahua,ground).
attr(chihuahua,fierce).
attr(chihuahua,timid).
attr(chihuahua,smart).
attr(chihuahua,solitary).
attr(chihuahua,domestic).

attr(rat,black).
attr(rat,white).
attr(rat,brown).
attr(rat,gray).
attr(rat,furry).
attr(rat,small).
attr(rat,bulbous).
attr(rat,lean).
attr(rat,paws).
attr(rat,tail).
attr(rat,meatteeth).
attr(rat,buckteeth).
attr(rat,claws).
attr(rat,smelly).
attr(rat,tunnels).
attr(rat,walks).
attr(rat,fast).
attr(rat,quadrapedal).
attr(rat,active).
attr(rat,nocturnal).
attr(rat,hibernate).
attr(rat,agility).
attr(rat,meat).
attr(rat,insects).
attr(rat,forager).
attr(rat,hunter).
attr(rat,scavenger).
attr(rat,newworld).
attr(rat,oldworld).
attr(rat,plains).
attr(rat,forest).
attr(rat,fields).
attr(rat,ground).
attr(rat,fierce).
attr(rat,smart).
attr(rat,solitary).
attr(rat,nestspot).

attr(weasel,black).
attr(weasel,brown).
attr(weasel,gray).
attr(weasel,furry).
attr(weasel,small).
attr(weasel,lean).
attr(weasel,paws).
attr(weasel,tail).
attr(weasel,chewteeth).
attr(weasel,meatteeth).
attr(weasel,claws).
attr(weasel,smelly).
attr(weasel,tunnels).
attr(weasel,walks).
attr(weasel,fast).
attr(weasel,muscle).
attr(weasel,quadrapedal).
attr(weasel,active).
attr(weasel,nocturnal).
attr(weasel,agility).
attr(weasel,meat).
attr(weasel,forager).
attr(weasel,hunter).
attr(weasel,newworld).
attr(weasel,oldworld).
attr(weasel,forest).
attr(weasel,fields).
attr(weasel,ground).
attr(weasel,fierce).
attr(weasel,smart).
attr(weasel,solitary).

attr(otter,black).
attr(otter,brown).
attr(otter,furry).
attr(otter,small).
attr(otter,bulbous).
attr(otter,lean).
attr(otter,flippers).
attr(otter,paws).
attr(otter,tail).
attr(otter,chewteeth).
attr(otter,meatteeth).
attr(otter,buckteeth).
attr(otter,claws).
attr(otter,swims).
attr(otter,fast).
attr(otter,quadrapedal).
attr(otter,active).
attr(otter,agility).
attr(otter,fish).
attr(otter,hunter).
attr(otter,newworld).
attr(otter,oldworld).
attr(otter,arctic).
attr(otter,coastal).
attr(otter,forest).
attr(otter,ocean).
attr(otter,water).
attr(otter,timid).
attr(otter,smart).
attr(otter,solitary).
attr(otter,nestspot).

attr(buffalo,black).
attr(buffalo,brown).
attr(buffalo,furry).
attr(buffalo,toughskin).
attr(buffalo,big).
attr(buffalo,bulbous).
attr(buffalo,hooves).
attr(buffalo,chewteeth).
attr(buffalo,horns).
attr(buffalo,smelly).
attr(buffalo,walks).
attr(buffalo,fast).
attr(buffalo,slow).
attr(buffalo,strong).
attr(buffalo,muscle).
attr(buffalo,quadrapedal).
attr(buffalo,inactive).
attr(buffalo,vegetation).
attr(buffalo,forager).
attr(buffalo,grazer).
attr(buffalo,newworld).
attr(buffalo,plains).
attr(buffalo,ground).
attr(buffalo,fierce).
attr(buffalo,group).

attr(zebra,black).
attr(zebra,white).
attr(zebra,stripes).
attr(zebra,furry).
attr(zebra,toughskin).
attr(zebra,big).
attr(zebra,lean).
attr(zebra,hooves).
attr(zebra,longleg).
attr(zebra,longneck).
attr(zebra,tail).
attr(zebra,chewteeth).
attr(zebra,walks).
attr(zebra,fast).
attr(zebra,strong).
attr(zebra,muscle).
attr(zebra,quadrapedal).
attr(zebra,active).
attr(zebra,agility).
attr(zebra,vegetation).
attr(zebra,forager).
attr(zebra,grazer).
attr(zebra,oldworld).
attr(zebra,bush).
attr(zebra,plains).
attr(zebra,fields).
attr(zebra,ground).
attr(zebra,timid).
attr(zebra,smart).
attr(zebra,group).

attr(giant_panda,black).
attr(giant_panda,white).
attr(giant_panda,patches).
attr(giant_panda,spots).
attr(giant_panda,furry).
attr(giant_panda,big).
attr(giant_panda,bulbous).
attr(giant_panda,pads).
attr(giant_panda,paws).
attr(giant_panda,chewteeth).
attr(giant_panda,buckteeth).
attr(giant_panda,claws).
attr(giant_panda,walks).
attr(giant_panda,slow).
attr(giant_panda,strong).
attr(giant_panda,bipedal).
attr(giant_panda,quadrapedal).
attr(giant_panda,inactive).
attr(giant_panda,fish).
attr(giant_panda,vegetation).
attr(giant_panda,forager).
attr(giant_panda,grazer).
attr(giant_panda,newworld).
attr(giant_panda,oldworld).
attr(giant_panda,bush).
attr(giant_panda,forest).
attr(giant_panda,jungle).
attr(giant_panda,ground).
attr(giant_panda,tree).
attr(giant_panda,timid).
attr(giant_panda,smart).
attr(giant_panda,group).
attr(giant_panda,solitary).
attr(giant_panda,nestspot).
attr(giant_panda,domestic).

attr(deer,brown).
attr(deer,patches).
attr(deer,spots).
attr(deer,furry).
attr(deer,big).
attr(deer,lean).
attr(deer,hooves).
attr(deer,longleg).
attr(deer,longneck).
attr(deer,tail).
attr(deer,chewteeth).
attr(deer,horns).
attr(deer,walks).
attr(deer,fast).
attr(deer,strong).
attr(deer,muscle).
attr(deer,quadrapedal).
attr(deer,active).
attr(deer,agility).
attr(deer,vegetation).
attr(deer,forager).
attr(deer,grazer).
attr(deer,newworld).
attr(deer,oldworld).
attr(deer,plains).
attr(deer,forest).
attr(deer,fields).
attr(deer,mountains).
attr(deer,ground).
attr(deer,timid).
attr(deer,smart).
attr(deer,group).
attr(deer,nestspot).

attr(bobcat,brown).
attr(bobcat,orange).
attr(bobcat,yellow).
attr(bobcat,spots).
attr(bobcat,furry).
attr(bobcat,small).
attr(bobcat,lean).
attr(bobcat,pads).
attr(bobcat,paws).
attr(bobcat,tail).
attr(bobcat,meatteeth).
attr(bobcat,claws).
attr(bobcat,walks).
attr(bobcat,fast).
attr(bobcat,strong).
attr(bobcat,muscle).
attr(bobcat,quadrapedal).
attr(bobcat,active).
attr(bobcat,hibernate).
attr(bobcat,agility).
attr(bobcat,meat).
attr(bobcat,forager).
attr(bobcat,hunter).
attr(bobcat,stalker).
attr(bobcat,newworld).
attr(bobcat,oldworld).
attr(bobcat,plains).
attr(bobcat,forest).
attr(bobcat,mountains).
attr(bobcat,ground).
attr(bobcat,tree).
attr(bobcat,cave).
attr(bobcat,fierce).
attr(bobcat,smart).
attr(bobcat,solitary).

attr(pig,black).
attr(pig,white).
attr(pig,brown).
attr(pig,gray).
attr(pig,patches).
attr(pig,spots).
attr(pig,hairless).
attr(pig,toughskin).
attr(pig,big).
attr(pig,bulbous).
attr(pig,hooves).
attr(pig,tail).
attr(pig,chewteeth).
attr(pig,smelly).
attr(pig,walks).
attr(pig,slow).
attr(pig,strong).
attr(pig,quadrapedal).
attr(pig,inactive).
attr(pig,vegetation).
attr(pig,forager).
attr(pig,newworld).
attr(pig,oldworld).
attr(pig,fields).
attr(pig,ground).
attr(pig,fierce).
attr(pig,timid).
attr(pig,smart).
attr(pig,group).
attr(pig,domestic).

attr(lion,brown).
attr(lion,yellow).
attr(lion,furry).
attr(lion,big).
attr(lion,bulbous).
attr(lion,lean).
attr(lion,pads).
attr(lion,paws).
attr(lion,tail).
attr(lion,meatteeth).
attr(lion,claws).
attr(lion,smelly).
attr(lion,walks).
attr(lion,fast).
attr(lion,strong).
attr(lion,muscle).
attr(lion,quadrapedal).
attr(lion,active).
attr(lion,inactive).
attr(lion,agility).
attr(lion,meat).
attr(lion,forager).
attr(lion,hunter).
attr(lion,stalker).
attr(lion,oldworld).
attr(lion,desert).
attr(lion,bush).
attr(lion,forest).
attr(lion,jungle).
attr(lion,mountains).
attr(lion,ground).
attr(lion,fierce).
attr(lion,smart).
attr(lion,group).
attr(lion,nestspot).

attr(mouse,white).
attr(mouse,brown).
attr(mouse,gray).
attr(mouse,furry).
attr(mouse,small).
attr(mouse,paws).
attr(mouse,tail).
attr(mouse,chewteeth).
attr(mouse,buckteeth).
attr(mouse,smelly).
attr(mouse,tunnels).
attr(mouse,walks).
attr(mouse,fast).
attr(mouse,weak).
attr(mouse,quadrapedal).
attr(mouse,active).
attr(mouse,nocturnal).
attr(mouse,hibernate).
attr(mouse,agility).
attr(mouse,vegetation).
attr(mouse,grazer).
attr(mouse,scavenger).
attr(mouse,newworld).
attr(mouse,oldworld).
attr(mouse,plains).
attr(mouse,forest).
attr(mouse,fields).
attr(mouse,ground).
attr(mouse,timid).
attr(mouse,group).
attr(mouse,nestspot).
attr(mouse,domestic).

attr(polar_bear,white).
attr(polar_bear,furry).
attr(polar_bear,toughskin).
attr(polar_bear,big).
attr(polar_bear,bulbous).
attr(polar_bear,pads).
attr(polar_bear,paws).
attr(polar_bear,chewteeth).
attr(polar_bear,meatteeth).
attr(polar_bear,claws).
attr(polar_bear,smelly).
attr(polar_bear,swims).
attr(polar_bear,walks).
attr(polar_bear,fast).
attr(polar_bear,slow).
attr(polar_bear,strong).
attr(polar_bear,bipedal).
attr(polar_bear,quadrapedal).
attr(polar_bear,active).
attr(polar_bear,inactive).
attr(polar_bear,hibernate).
attr(polar_bear,agility).
attr(polar_bear,fish).
attr(polar_bear,meat).
attr(polar_bear,forager).
attr(polar_bear,hunter).
attr(polar_bear,scavenger).
attr(polar_bear,stalker).
attr(polar_bear,newworld).
attr(polar_bear,oldworld).
attr(polar_bear,arctic).
attr(polar_bear,coastal).
attr(polar_bear,ocean).
attr(polar_bear,ground).
attr(polar_bear,water).
attr(polar_bear,fierce).
attr(polar_bear,solitary).

attr(collie,white).
attr(collie,brown).
attr(collie,patches).
attr(collie,furry).
attr(collie,big).
attr(collie,small).
attr(collie,lean).
attr(collie,paws).
attr(collie,tail).
attr(collie,chewteeth).
attr(collie,meatteeth).
attr(collie,walks).
attr(collie,fast).
attr(collie,quadrapedal).
attr(collie,active).
attr(collie,agility).
attr(collie,meat).
attr(collie,hunter).
attr(collie,newworld).
attr(collie,oldworld).
attr(collie,ground).
attr(collie,timid).
attr(collie,smart).
attr(collie,solitary).
attr(collie,domestic).

attr(walrus,brown).
attr(walrus,gray).
attr(walrus,hairless).
attr(walrus,toughskin).
attr(walrus,big).
attr(walrus,bulbous).
attr(walrus,flippers).
attr(walrus,chewteeth).
attr(walrus,meatteeth).
attr(walrus,buckteeth).
attr(walrus,strainteeth).
attr(walrus,tusks).
attr(walrus,smelly).
attr(walrus,swims).
attr(walrus,fast).
attr(walrus,slow).
attr(walrus,strong).
attr(walrus,inactive).
attr(walrus,fish).
attr(walrus,newworld).
attr(walrus,arctic).
attr(walrus,coastal).
attr(walrus,ocean).
attr(walrus,water).
attr(walrus,timid).
attr(walrus,smart).
attr(walrus,group).
attr(walrus,nestspot).

attr(raccoon,black).
attr(raccoon,white).
attr(raccoon,gray).
attr(raccoon,patches).
attr(raccoon,spots).
attr(raccoon,stripes).
attr(raccoon,furry).
attr(raccoon,small).
attr(raccoon,pads).
attr(raccoon,paws).
attr(raccoon,tail).
attr(raccoon,chewteeth).
attr(raccoon,meatteeth).
attr(raccoon,claws).
attr(raccoon,walks).
attr(raccoon,fast).
attr(raccoon,quadrapedal).
attr(raccoon,active).
attr(raccoon,nocturnal).
attr(raccoon,hibernate).
attr(raccoon,agility).
attr(raccoon,fish).
attr(raccoon,meat).
attr(raccoon,vegetation).
attr(raccoon,forager).
attr(raccoon,scavenger).
attr(raccoon,newworld).
attr(raccoon,oldworld).
attr(raccoon,forest).
attr(raccoon,fields).
attr(raccoon,mountains).
attr(raccoon,ground).
attr(raccoon,tree).
attr(raccoon,fierce).
attr(raccoon,smart).
attr(raccoon,solitary).
attr(raccoon,nestspot).

attr(cow,black).
attr(cow,white).
attr(cow,brown).
attr(cow,patches).
attr(cow,spots).
attr(cow,furry).
attr(cow,toughskin).
attr(cow,big).
attr(cow,bulbous).
attr(cow,hooves).
attr(cow,tail).
attr(cow,chewteeth).
attr(cow,horns).
attr(cow,smelly).
attr(cow,walks).
attr(cow,slow).
attr(cow,strong).
attr(cow,quadrapedal).
attr(cow,active).
attr(cow,inactive).
attr(cow,vegetation).
attr(cow,grazer).
attr(cow,newworld).
attr(cow,oldworld).
attr(cow,plains).
attr(cow,fields).
attr(cow,ground).
attr(cow,timid).
attr(cow,group).
attr(cow,domestic).

attr(dolphin,white).
attr(dolphin,blue).
attr(dolphin,gray).
attr(dolphin,hairless).
attr(dolphin,toughskin).
attr(dolphin,big).
attr(dolphin,lean).
attr(dolphin,flippers).
attr(dolphin,tail).
attr(dolphin,chewteeth).
attr(dolphin,swims).
attr(dolphin,fast).
attr(dolphin,strong).
attr(dolphin,muscle).
attr(dolphin,active).
attr(dolphin,agility).
attr(dolphin,fish).
attr(dolphin,newworld).
attr(dolphin,oldworld).
attr(dolphin,coastal).
attr(dolphin,ocean).
attr(dolphin,water).
attr(dolphin,timid).
attr(dolphin,smart).
attr(dolphin,group).
attr(dolphin,domestic).



% -------------------------
% Implication hierarchy
% -------------------------
% -------------------------
% HABITAT / LOCATION
% -------------------------
implies(ocean,water).
implies(coastal,water).
implies(arctic,water).

implies(coastal,ocean).        % coastal implies ocean-ish environment
implies(arctic,oldworld).      % optional: if you want arctic to count as "oldworld" too (tune as desired)
% If you *don't* want this kind of geo inference, remove it.

implies(forest,ground).
implies(fields,ground).
implies(plains,ground).
implies(jungle,ground).
implies(desert,ground).
implies(mountains,ground).
implies(bush,ground).
implies(cave,ground).
implies(tree,ground).
implies(tunnels,ground).
implies(nestspot,ground).

implies(jungle,forest).        % jungle is a kind of forest habitat
implies(bush,fields).          % bushland behaves like open fields for many purposes
implies(plains,fields).        % plains are a kind of open field

% -------------------------
% MOVEMENT / BODY PLAN
% -------------------------
implies(swims,moves).
implies(walks,moves).
implies(flys,moves).
implies(hops,moves).

implies(bipedal,walks).        % if bipedal then walking-capable
implies(quadrapedal,walks).    % if quadrupedal then walking-capable

% -------------------------
% DIET / FEEDING
% -------------------------
implies(fish,meat).
implies(insects,meat).
implies(meat,food).
implies(vegetation,food).
implies(plankton,food).

implies(grazer,forager).       % grazing is a kind of foraging
implies(skimmer,forager).      % skimming is a kind of foraging

implies(hunter,forager).       % hunting is a kind of foraging strategy
implies(scavenger,forager).    % scavenging is a kind of foraging
implies(stalker,hunter).       % stalking is a kind of hunting

% If you want: "eats fish -> hunter" is not always true (filter-feeders exist),
% so I’m not forcing it.

% -------------------------
% TEETH / MOUTH PARTS
% -------------------------
implies(meatteeth,chewteeth).
implies(buckteeth,chewteeth).
implies(strainteeth,chewteeth).

% -------------------------
% LIMBS / APPENDAGES
% -------------------------
implies(paws,limbs).
implies(hands,limbs).
implies(flippers,limbs).
implies(hooves,limbs).

implies(pads,paws).            % pads implies paws-like feet

% -------------------------
% WEAPONS / DEFENSE
% -------------------------
implies(claws,sharp).
implies(tusks,sharp).
implies(horns,sharp).

implies(sharp,dangerous).      % general high-level concept

% -------------------------
% SIZE / SHAPE
% -------------------------
implies(big,size).
implies(small,size).

implies(lean,bodyshape).
implies(bulbous,bodyshape).

% -------------------------
% COAT / SKIN
% -------------------------
implies(furry,has_covering).
implies(hairless,has_covering).  % still a "covering" state

implies(toughskin,has_covering).

% -------------------------
% COLORS / PATTERNS
% -------------------------
implies(spots,pattern).
implies(stripes,pattern).
implies(patches,pattern).

implies(black,color).
implies(white,color).
implies(brown,color).
implies(gray,color).
implies(blue,color).
implies(orange,color).
implies(yellow,color).
implies(red,color).

% -------------------------
% TEMPERAMENT / SOCIAL
% -------------------------
implies(group,social).
implies(solitary,social).

implies(fierce,temperament).
implies(timid,temperament).

% -------------------------
% ACTIVITY / TIME
% -------------------------
implies(nocturnal,active_time).
implies(hibernate,inactive).   % hibernation implies inactive (you already use inactive)
implies(active,not_inactive).  % NOTE: only safe if you do NOT use strong negation; otherwise omit
% If DePa doesn’t like "not_inactive" as a derived atom, delete this line.

% -------------------------
% OTHER SENSORY / TRAITS
% -------------------------
implies(smelly,trait).
implies(smart,trait).
implies(strong,trait).
implies(weak,trait).
implies(muscle,trait).
implies(fast,trait).
implies(slow,trait).

% -------------------------
% DOMESTICATION / WORLD
% -------------------------
implies(domestic,managed).
implies(newworld,world).
implies(oldworld,world).




% -------------------------
% Attribute closure (normal rules)
% -------------------------
has_attr(X) :- sel(A), attr(A,X).
has_attr(Y) :- has_attr(X), implies(X,Y).

% -------------------------
% Output
% -------------------------
class(A) :- sel(A).
attributes(X) :- has_attr(X).

#show class/1.
#show attributes/1.