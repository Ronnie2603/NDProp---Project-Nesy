class(antelope) :- bad.
class(grizzly_bear) :- bad.
class(killer_whale) :- bad.
class(beaver) :- bad.
class(dalmatian) :- bad.
class(persian_cat) :- attr(furry), attr(pads), attr(paws), attr(chewteeth), attr(meatteeth), attr(claws), attr(quadrapedal), attr(domestic), nattr(hairless), nattr(toughskin), nattr(flippers), nattr(hands), nattr(hooves), nattr(buckteeth), nattr(horns), nattr(tusks), nattr(bipedal).
class(horse) :- bad.
class(german_shepherd) :- bad.
class(blue_whale) :- bad.
class(siamese_cat) :- bad.
class(skunk) :- bad.
class(mole) :- bad.
class(tiger) :- bad.
class(hippopotamus) :- attr(hairless), attr(toughskin), attr(chewteeth), attr(quadrapedal), nattr(furry), nattr(flippers), nattr(hands), nattr(hooves), nattr(pads), nattr(paws), nattr(meatteeth), nattr(buckteeth), nattr(horns), nattr(claws), nattr(tusks), nattr(bipedal), nattr(domestic).
class(leopard) :- attr(furry), attr(paws), attr(meatteeth), attr(claws), attr(quadrapedal), nattr(hairless), nattr(toughskin), nattr(flippers), nattr(hands), nattr(hooves), nattr(pads), nattr(chewteeth), nattr(buckteeth), nattr(horns), nattr(tusks), nattr(bipedal), nattr(domestic).
class(moose) :- bad.
class(spider_monkey) :- bad.
class(humpback_whale) :- attr(hairless), attr(toughskin), attr(flippers), nattr(furry), nattr(hands), nattr(hooves), nattr(pads), nattr(paws), nattr(chewteeth), nattr(meatteeth), nattr(buckteeth), nattr(horns), nattr(claws), nattr(tusks), nattr(bipedal), nattr(quadrapedal), nattr(domestic).
class(elephant) :- bad.
class(gorilla) :- bad.
class(ox) :- bad.
class(fox) :- bad.
class(sheep) :- bad.
class(seal) :- attr(hairless), attr(toughskin), attr(flippers), attr(domestic), nattr(furry), nattr(hands), nattr(hooves), nattr(pads), nattr(paws), nattr(chewteeth), nattr(meatteeth), nattr(buckteeth), nattr(horns), nattr(claws), nattr(tusks), nattr(bipedal), nattr(quadrapedal).
class(chimpanzee) :- attr(furry), attr(toughskin), attr(hands), attr(chewteeth), attr(meatteeth), attr(bipedal), attr(quadrapedal), attr(domestic), nattr(hairless), nattr(flippers), nattr(hooves), nattr(pads), nattr(paws), nattr(buckteeth), nattr(horns), nattr(claws), nattr(tusks).
class(hamster) :- bad.
class(squirrel) :- bad.
class(rhinoceros) :- bad.
class(rabbit) :- bad.
class(bat) :- bad.
class(giraffe) :- bad.
class(wolf) :- bad.
class(chihuahua) :- bad.
class(rat) :- attr(furry), attr(paws), attr(meatteeth), attr(buckteeth), attr(claws), attr(quadrapedal), nattr(hairless), nattr(toughskin), nattr(flippers), nattr(hands), nattr(hooves), nattr(pads), nattr(chewteeth), nattr(horns), nattr(tusks), nattr(bipedal), nattr(domestic).
class(weasel) :- bad.
class(otter) :- bad.
class(buffalo) :- bad.
class(zebra) :- bad.
class(giant_panda) :- attr(furry), attr(pads), attr(paws), attr(chewteeth), attr(buckteeth), attr(claws), attr(bipedal), attr(quadrapedal), attr(domestic), nattr(hairless), nattr(toughskin), nattr(flippers), nattr(hands), nattr(hooves), nattr(meatteeth), nattr(horns), nattr(tusks).
class(deer) :- bad.
class(bobcat) :- bad.
class(pig) :- attr(hairless), attr(toughskin), attr(hooves), attr(chewteeth), attr(quadrapedal), attr(domestic), nattr(furry), nattr(flippers), nattr(hands), nattr(pads), nattr(paws), nattr(meatteeth), nattr(buckteeth), nattr(horns), nattr(claws), nattr(tusks), nattr(bipedal).
class(lion) :- bad.
class(mouse) :- bad.
class(polar_bear) :- bad.
class(collie) :- bad.
class(walrus) :- bad.
class(raccoon) :- attr(furry), attr(pads), attr(paws), attr(chewteeth), attr(meatteeth), attr(claws), attr(quadrapedal), nattr(hairless), nattr(toughskin), nattr(flippers), nattr(hands), nattr(hooves), nattr(buckteeth), nattr(horns), nattr(tusks), nattr(bipedal), nattr(domestic).
class(cow) :- bad.
class(dolphin) :- bad.
bad :- not bad.
class(persian_cat) :- not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(hippopotamus) :- not class(persian_cat), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(leopard) :- not class(persian_cat), not class(hippopotamus), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(humpback_whale) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(seal) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(chimpanzee) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(rat), not class(giant_panda), not class(pig), not class(raccoon).
class(rat) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(giant_panda), not class(pig), not class(raccoon).
class(giant_panda) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(pig), not class(raccoon).
class(pig) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(raccoon).
class(raccoon) :- not class(persian_cat), not class(hippopotamus), not class(leopard), not class(humpback_whale), not class(seal), not class(chimpanzee), not class(rat), not class(giant_panda), not class(pig).
attr(black) :- not nattr(black).
nattr(black) :- not attr(black).
attr(white) :- not nattr(white).
nattr(white) :- not attr(white).
attr(blue) :- not nattr(blue).
nattr(blue) :- not attr(blue).
attr(brown) :- not nattr(brown).
nattr(brown) :- not attr(brown).
attr(gray) :- not nattr(gray).
nattr(gray) :- not attr(gray).
attr(orange) :- not nattr(orange).
nattr(orange) :- not attr(orange).
attr(red) :- not nattr(red).
nattr(red) :- not attr(red).
attr(yellow) :- not nattr(yellow).
nattr(yellow) :- not attr(yellow).
attr(patches) :- not nattr(patches).
nattr(patches) :- not attr(patches).
attr(spots) :- not nattr(spots).
nattr(spots) :- not attr(spots).
attr(stripes) :- not nattr(stripes).
nattr(stripes) :- not attr(stripes).
attr(furry) :- not nattr(furry).
nattr(furry) :- not attr(furry).
attr(hairless) :- not nattr(hairless).
nattr(hairless) :- not attr(hairless).
attr(toughskin) :- not nattr(toughskin).
nattr(toughskin) :- not attr(toughskin).
attr(big) :- not nattr(big).
nattr(big) :- not attr(big).
attr(small) :- not nattr(small).
nattr(small) :- not attr(small).
attr(bulbous) :- not nattr(bulbous).
nattr(bulbous) :- not attr(bulbous).
attr(lean) :- not nattr(lean).
nattr(lean) :- not attr(lean).
attr(flippers) :- not nattr(flippers).
nattr(flippers) :- not attr(flippers).
attr(hands) :- not nattr(hands).
nattr(hands) :- not attr(hands).
attr(hooves) :- not nattr(hooves).
nattr(hooves) :- not attr(hooves).
attr(pads) :- not nattr(pads).
nattr(pads) :- not attr(pads).
attr(paws) :- not nattr(paws).
nattr(paws) :- not attr(paws).
attr(longleg) :- not nattr(longleg).
nattr(longleg) :- not attr(longleg).
attr(longneck) :- not nattr(longneck).
nattr(longneck) :- not attr(longneck).
attr(tail) :- not nattr(tail).
nattr(tail) :- not attr(tail).
attr(chewteeth) :- not nattr(chewteeth).
nattr(chewteeth) :- not attr(chewteeth).
attr(meatteeth) :- not nattr(meatteeth).
nattr(meatteeth) :- not attr(meatteeth).
attr(buckteeth) :- not nattr(buckteeth).
nattr(buckteeth) :- not attr(buckteeth).
attr(strainteeth) :- not nattr(strainteeth).
nattr(strainteeth) :- not attr(strainteeth).
attr(horns) :- not nattr(horns).
nattr(horns) :- not attr(horns).
attr(claws) :- not nattr(claws).
nattr(claws) :- not attr(claws).
attr(tusks) :- not nattr(tusks).
nattr(tusks) :- not attr(tusks).
attr(smelly) :- not nattr(smelly).
nattr(smelly) :- not attr(smelly).
attr(flys) :- not nattr(flys).
nattr(flys) :- not attr(flys).
attr(hops) :- not nattr(hops).
nattr(hops) :- not attr(hops).
attr(swims) :- not nattr(swims).
nattr(swims) :- not attr(swims).
attr(tunnels) :- not nattr(tunnels).
nattr(tunnels) :- not attr(tunnels).
attr(walks) :- not nattr(walks).
nattr(walks) :- not attr(walks).
attr(fast) :- not nattr(fast).
nattr(fast) :- not attr(fast).
attr(slow) :- not nattr(slow).
nattr(slow) :- not attr(slow).
attr(strong) :- not nattr(strong).
nattr(strong) :- not attr(strong).
attr(weak) :- not nattr(weak).
nattr(weak) :- not attr(weak).
attr(muscle) :- not nattr(muscle).
nattr(muscle) :- not attr(muscle).
attr(bipedal) :- not nattr(bipedal).
nattr(bipedal) :- not attr(bipedal).
attr(quadrapedal) :- not nattr(quadrapedal).
nattr(quadrapedal) :- not attr(quadrapedal).
attr(active) :- not nattr(active).
nattr(active) :- not attr(active).
attr(inactive) :- not nattr(inactive).
nattr(inactive) :- not attr(inactive).
attr(nocturnal) :- not nattr(nocturnal).
nattr(nocturnal) :- not attr(nocturnal).
attr(hibernate) :- not nattr(hibernate).
nattr(hibernate) :- not attr(hibernate).
attr(agility) :- not nattr(agility).
nattr(agility) :- not attr(agility).
attr(fish) :- not nattr(fish).
nattr(fish) :- not attr(fish).
attr(meat) :- not nattr(meat).
nattr(meat) :- not attr(meat).
attr(plankton) :- not nattr(plankton).
nattr(plankton) :- not attr(plankton).
attr(vegetation) :- not nattr(vegetation).
nattr(vegetation) :- not attr(vegetation).
attr(insects) :- not nattr(insects).
nattr(insects) :- not attr(insects).
attr(forager) :- not nattr(forager).
nattr(forager) :- not attr(forager).
attr(grazer) :- not nattr(grazer).
nattr(grazer) :- not attr(grazer).
attr(hunter) :- not nattr(hunter).
nattr(hunter) :- not attr(hunter).
attr(scavenger) :- not nattr(scavenger).
nattr(scavenger) :- not attr(scavenger).
attr(skimmer) :- not nattr(skimmer).
nattr(skimmer) :- not attr(skimmer).
attr(stalker) :- not nattr(stalker).
nattr(stalker) :- not attr(stalker).
attr(newworld) :- not nattr(newworld).
nattr(newworld) :- not attr(newworld).
attr(oldworld) :- not nattr(oldworld).
nattr(oldworld) :- not attr(oldworld).
attr(arctic) :- not nattr(arctic).
nattr(arctic) :- not attr(arctic).
attr(coastal) :- not nattr(coastal).
nattr(coastal) :- not attr(coastal).
attr(desert) :- not nattr(desert).
nattr(desert) :- not attr(desert).
attr(bush) :- not nattr(bush).
nattr(bush) :- not attr(bush).
attr(plains) :- not nattr(plains).
nattr(plains) :- not attr(plains).
attr(forest) :- not nattr(forest).
nattr(forest) :- not attr(forest).
attr(fields) :- not nattr(fields).
nattr(fields) :- not attr(fields).
attr(jungle) :- not nattr(jungle).
nattr(jungle) :- not attr(jungle).
attr(mountains) :- not nattr(mountains).
nattr(mountains) :- not attr(mountains).
attr(ocean) :- not nattr(ocean).
nattr(ocean) :- not attr(ocean).
attr(ground) :- not nattr(ground).
nattr(ground) :- not attr(ground).
attr(water) :- not nattr(water).
nattr(water) :- not attr(water).
attr(tree) :- not nattr(tree).
nattr(tree) :- not attr(tree).
attr(cave) :- not nattr(cave).
nattr(cave) :- not attr(cave).
attr(fierce) :- not nattr(fierce).
nattr(fierce) :- not attr(fierce).
attr(timid) :- not nattr(timid).
nattr(timid) :- not attr(timid).
attr(smart) :- not nattr(smart).
nattr(smart) :- not attr(smart).
attr(group) :- not nattr(group).
nattr(group) :- not attr(group).
attr(solitary) :- not nattr(solitary).
nattr(solitary) :- not attr(solitary).
attr(nestspot) :- not nattr(nestspot).
nattr(nestspot) :- not attr(nestspot).
attr(domestic) :- not nattr(domestic).
nattr(domestic) :- not attr(domestic).

attr(fish) :- attr(ocean).
attr(water) :- attr(swims).
nattr(swims) :- nattr(water).
attr(swims) :- attr(flippers).

class(lion) :- attr(desert).

attr(water) :- attr(ocean).
attr(swims) :- attr(water).
attr(swims) :- attr(ocean).
attr(fish) :- attr(swims).
attr(meatteeth) :- attr(meat).
attr(quadrapedal) :- attr(hooves).
attr(quadrapedal) :- attr(paws).
attr(bipedal) :- attr(hands).
attr(ground) :- attr(desert).
attr(ground) :- attr(plains).
attr(ground) :- attr(fields).
attr(ground) :- attr(bush).

