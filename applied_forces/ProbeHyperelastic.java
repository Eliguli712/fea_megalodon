import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeHyperelastic {
  public static Model run() {
    Model model = ModelUtil.create("Probe");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").run();
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").automatic(true);
    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    model.component("comp1").physics("solid").create("hmm1", "HyperelasticModel", 3);
    model.component("comp1").physics("solid").feature("hmm1").label("Hyperelastic Material Probe");

    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.study("std1").feature("stat").activate("solid", true);

    try {
      model.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/probe_hyperelastic.mph");
    } catch (IOException e) {
      throw new RuntimeException("Failed to save probe model", e);
    }
    return model;
  }
  public static void main(String[] args) { run(); }
}
