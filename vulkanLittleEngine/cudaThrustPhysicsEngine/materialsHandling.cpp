#include <unordered_map>
#include <string>
#include <array>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "materials.cuh"

namespace MLE::MLPE::MATERIALS {

	typedef std::unordered_map< std::string, std::vector<rbp::massElement> > massDistributionHash;

	std::array<std::string, N> materialNames = 
	{
		"Hydrogen", "Helium", "Lithium", "Berylium",
		"Boron", "Carbon", "Nitrogen", "Oxygen",
		"Fluorine", "Neon", "Sodium", "Magnesium",
		"Aluminium", "Silicon", "Phosphorus", "Sulfur",
		"Chlorine", "Argon", "Potassium", "Calcium",
		"Scandium", "Titanium", "Vanadium", "Chromium",
		"Manganese", "Iron", "Cobalt", "Nickel",
		"Copper", "Zinc", "Gallium", "Germanium",
		"Arsenic", "Selenium", "Bromine", "Krypton",
		"Rubidium", "Stronium", "Yttrium", "Zirconium",
		"Niobium", "Molybdenum", "Technetium", "Ruthenium",
		"Rhodium", "Palladium", "Silver", "Cadmium",
		"Indium", "Tin", "Antimony", "Tellurium",
		"Iodine", "Xenon", "Cesium", "Barium",
		"Lanthanum", "Cerium", "Praseodymium", "Neodymium",
		"Promethium", "Samarium", "Europium", "Gadolinium",
		"Terbium", "Dysprosium", "Holmium", "Erbium",
		"Thulium", "Ytterbium", "Lutetium", "Hafnium",
		"Tantalum", "Tungsten", "Rhenium", "Osmium",
		"Iridium", "Platinum", "Gold", "Mercury", 
		"Thallium", "Lead", "Bismuth", "Polonium", 
		"Astatine", "Radon", "Francium", "Radium",
		"Actinium", "Thorium", "Protactinium", "Uranium",
		"Neptunium", "Plutonium", "Americium", "Curium",
		"Berklium", "Californium", "Einsteinium", "Fermium",
		"Mendelevium", "Nobelium", "Lawrencium", "Rutherfordium",
		"Dubnium", "Seaborgium", "Bohrium", "Hassium", 
		"Meitnerium", " Darmstadtium", "Roentgenium", "Copernicium",
		"Nhonium", "Flerovium", "Moscovium", "Livermorium", 
		"Tennessine", "Oganesson"
	};

	std::vector<float> massArrayFromStruct(std::vector<rbp::massElement> massElements) {
		std::vector<float> matrixElements(massElements.size());
		thrust::transform(
			thrust::device,
			thrust::device_pointer_cast(massElements.data()),
			thrust::device_pointer_cast(massElements.data()) + massElements.size(),
			thrust::device_pointer_cast(matrixElements.data()),
			getFirst());

		return matrixElements;
	}

	template<typename T>
	std::unique_ptr<T[]> timeDependentMatrixCreate(std::vector<T> matrixElements, T dt) {
		T timeFactor = 1 / (dt * dt);
		int MS = static_cast<int>(matrixElements.size());

		std::unique_ptr<T[]> diagMassMatrix = std::make_unique<T[]>(MS * MS);
		//T* diagMassMatrix[NxN];

		for (int i = 0; i < MS; i++)for (int j = 0; j < MS; j++) {
			*(diagMassMatrix + j * MS + i) = !(i - j) ? timeFactor * matrixElements[i] : 0;
		}

		return diagMassMatrix;
	}

	// DM
	void MATERIAL_BUILD::buildMaterialDampingMatrix() {

	}

	// Mh
	void MATERIAL_BUILD::buildMaterialMassDistributionMatrix(std::vector<rbp::massElement> massElements) {
		auto massArray = massArrayFromStruct(massElements);
		material->materialMassDistributionMatrix = timeDependentMatrixCreate<float>(massArray, (float)DT).release();
	}

	// Kd
	void MATERIAL_BUILD::buildLocalDiagonalDampingNeighborhoodMatrix() {

	}

	// Kq
	void MATERIAL_BUILD::buildLocalDampingNeighborhoodMatrix() {

	}

	void MATERIAL_BUILD::buildHeatDeformationMatrix() {

	}

	void MATERIAL_BUILD::buildLocalHeatFluctuationDistribution() {

	}

	void MATERIAL_BUILD::buildMaterial(std::string type , std::vector<rbp::massElement> massElements) {
		buildMaterialDampingMatrix();
		buildMaterialMassDistributionMatrix(massElements);
		buildLocalDiagonalDampingNeighborhoodMatrix();
		buildLocalDampingNeighborhoodMatrix();
		buildHeatDeformationMatrix();
		buildLocalHeatFluctuationDistribution();
	}

	void MATERIAL_BUILD::destroyMaterialStruct() {
		delete material;
	}

	void buildMaterialList(massDistributionHash densityHash) {
		for (const auto& name : materialNames) {
			MATERIAL_BUILD element{ name, densityHash[name]};
			materials[name] = *(element.material);
		}
	}
}