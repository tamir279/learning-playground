#pragma once
#include <unordered_map>
#include <string>
#include <array>
#include "physicsEngine.cuh"

#define N 118

namespace MLE::MLPE {
	namespace MATERIALS {

		/*
		EXPLENATION: the massElement arrays represent the ordinary mass distribution of a particular 
		element in the periodic table, basically it represent material density
		*/

		// helpers
		typedef std::unordered_map< std::string, std::vector<rbp::massElement> > massDistributionHash;

		template<typename T>
		std::unique_ptr<T[]> timeDependentMatrixCreate(std::vector<T> matrixElements, T dt);

		struct getFirst : thrust::unary_function<rbp::massElement, float> {
			__host__ __device__ float operator()(rbp::massElement mE) {
				return mE.m;
			}
		};

		// define a probibility distribution from set of std pre-made distributions
		struct p_distribution {

		};

		// ***** PART 1 : DEFINE A MATERIAL *****
		struct MATERIAL {
			const float* materialDampingMatrix;
			const float* materialMassDistributionMatrix;
			const float* localDiagonalDampingNeighborhoodMatrix;
			const float* localDampingNeighborhoodMatrix;
			const float* heatDeformationMatrix;
			const p_distribution* localHeatFluctuationDistribution;
		};

		class MATERIAL_BUILD {
		public:

			MATERIAL* material;

			MATERIAL_BUILD(std::string type, std::vector<rbp::massElement> massElements) {
				buildMaterial(type, massElements);
			}

			~MATERIAL_BUILD() {
				destroyMaterialStruct();
			}

		private:
			void buildMaterial(std::string type, std::vector<rbp::massElement> massElements);
			void buildMaterialDampingMatrix();
			void buildMaterialMassDistributionMatrix(std::vector<rbp::massElement> massElements);
			void buildLocalDiagonalDampingNeighborhoodMatrix();
			void buildLocalDampingNeighborhoodMatrix();
			void buildHeatDeformationMatrix();
			void buildLocalHeatFluctuationDistribution();
			void destroyMaterialStruct();
		};

		// ***** PART 2 : SET A MAP OF SEVERAL MATERIALS *****

		std::unordered_map<std::string, MATERIAL> materials;
		// ***** PART 3 : POPULATE WITH MULTIPLE MATERIALS *****
		/*
		the current material map contains all known chemical elements
		*/
		void buildMaterialList(massDistributionHash densityHash);

	}
}