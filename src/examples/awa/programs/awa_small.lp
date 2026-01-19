% --- core idea: pick 2–3 “signature” attrs, then exclude the closest confounders with 'not'

class(antelope)        :- attr(hooves), attr(horns), not attr(domestic).

class(grizzly_bear)    :- attr(hibernate), attr(claws), not attr(white).

class(killer_whale)    :- attr(swims), attr(patches), not attr(furry).

class(beaver)          :- attr(buckteeth), attr(swims), not attr(flippers).

class(dalmatian)       :- attr(spots), attr(domestic), not attr(claws).

class(persian_cat)     :- attr(blue), attr(domestic), not attr(active).

class(horse)           :- attr(hooves), attr(longneck), attr(domestic).

class(german_shepherd) :- attr(smelly), attr(stalker), attr(domestic).

class(blue_whale)      :- attr(skimmer), attr(spots), not attr(chewteeth).

class(siamese_cat)     :- attr(nocturnal), attr(domestic), not attr(blue).

class(skunk)           :- attr(stripes), attr(smelly), not attr(hooves).

class(mole)            :- attr(tunnels), attr(buckteeth), not attr(hops).

class(tiger)           :- attr(orange), attr(stripes).

class(hippopotamus)    :- attr(hairless), attr(swims), not attr(flippers).

class(leopard)         :- attr(spots), attr(claws), not attr(stripes).

class(moose)           :- attr(horns), attr(longneck), not attr(domestic).

class(spider_monkey)   :- attr(hands), attr(tail), not attr(hooves).

class(humpback_whale)  :- attr(skimmer), attr(coastal), not attr(spots).

class(elephant)        :- attr(tusks), not attr(horns).

class(gorilla)         :- attr(hands), not attr(tail).

class(ox)              :- attr(horns), attr(domestic), not attr(agility).

class(fox)             :- attr(red), not attr(domestic).

class(sheep)           :- attr(hooves), attr(domestic), not attr(strong).

class(seal)            :- attr(flippers), not attr(chewteeth).

class(chimpanzee)      :- attr(insects), attr(hands), not attr(domestic).

class(hamster)         :- attr(hops), attr(domestic), not attr(swims).

class(squirrel)        :- attr(tree), attr(hops), not attr(domestic).

class(rhinoceros)      :- attr(horns), attr(toughskin), not attr(furry).

class(rabbit)          :- attr(hops), attr(buckteeth), not attr(meatteeth).

class(bat)             :- attr(flys), not attr(swims).

class(giraffe)         :- attr(longneck), not attr(horns).

class(wolf)            :- attr(cave), attr(meatteeth), not attr(domestic).

class(chihuahua)       :- attr(domestic), attr(small), not attr(buckteeth).

class(rat)             :- attr(tunnels), attr(meatteeth), not attr(hops).

class(weasel)          :- attr(tunnels), attr(meatteeth), not attr(buckteeth).

class(otter)           :- attr(swims), attr(buckteeth), not attr(skimmer).

class(buffalo)         :- attr(horns), attr(fierce), not attr(domestic).

class(zebra)           :- attr(stripes), attr(hooves), not attr(domestic).

class(giant_panda)     :- attr(bipedal), attr(buckteeth), not attr(stripes).

class(deer)            :- attr(horns), not attr(domestic).

class(bobcat)          :- attr(meatteeth), attr(spots), not attr(big).

class(pig)             :- attr(hairless), attr(domestic), not attr(horns).

class(lion)            :- attr(yellow), not attr(stripes).

class(mouse)           :- attr(scavenger), attr(buckteeth), not attr(meatteeth).

class(polar_bear)      :- attr(white), attr(meatteeth), not attr(patches).

class(collie)          :- attr(domestic), attr(timid), not attr(spots).

class(walrus)          :- attr(tusks), not attr(chewteeth).

class(raccoon)         :- attr(scavenger), attr(stripes), not attr(skunk).

class(cow)             :- attr(domestic), attr(horns), not attr(solitary).

class(dolphin)         :- attr(swims), attr(chewteeth), not attr(furry).

1 { attr(black); attr(white); attr(blue); attr(brown); attr(gray); attr(orange); attr(red); attr(yellow); attr(patches); attr(spots); attr(stripes); attr(furry); attr(hairless); attr(toughskin); attr(big); attr(small); attr(bulbous); attr(lean); attr(flippers); attr(hands); attr(hooves); attr(pads); attr(paws); attr(longleg); attr(longneck); attr(tail); attr(chewteeth); attr(meatteeth); attr(buckteeth); attr(strainteeth); attr(horns); attr(claws); attr(tusks); attr(smelly); attr(flys); attr(hops); attr(swims); attr(tunnels); attr(walks); attr(fast); attr(slow); attr(strong); attr(weak); attr(muscle); attr(bipedal); attr(quadrapedal); attr(active); attr(inactive); attr(nocturnal); attr(hibernate); attr(agility); attr(fish); attr(meat); attr(plankton); attr(vegetation); attr(insects); attr(forager); attr(grazer); attr(hunter); attr(scavenger); attr(skimmer); attr(stalker); attr(newworld); attr(oldworld); attr(arctic); attr(coastal); attr(desert); attr(bush); attr(plains); attr(forest); attr(fields); attr(jungle); attr(mountains); attr(ocean); attr(ground); attr(water); attr(tree); attr(cave); attr(fierce); attr(timid); attr(smart); attr(group); attr(solitary); attr(nestspot); attr(domestic) } 85.
