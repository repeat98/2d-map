import React, { useState, useMemo } from 'react';
import './FilterPanel.scss';

const FilterCategory = ({ title, options, activeItems, onToggle }) => {
  const [searchTerm, setSearchTerm] = useState('');

  // Filter options based on search term
  const filteredOptions = useMemo(() => {
    if (!options || options.length === 0) return [];
    if (!searchTerm.trim()) return options;
    
    const lowerSearchTerm = searchTerm.toLowerCase().trim();
    return options.filter(option => 
      option.name.toLowerCase().includes(lowerSearchTerm)
    );
  }, [options, searchTerm]);

  if (!options || options.length === 0) return null;

  return (
    <div className="FilterCategory">
      <h4 className="FilterCategoryTitle">{title}</h4>
      <div className="FilterSearchContainer">
        <input
          type="text"
          placeholder={`Search ${title.toLowerCase()}...`}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="FilterSearchInput"
        />
        {searchTerm && (
          <button
            onClick={() => setSearchTerm('')}
            className="FilterSearchClear"
            aria-label="Clear search"
          >
            ×
          </button>
        )}
      </div>
      <ul className="FilterCheckboxList">
        {filteredOptions.length > 0 ? (
          filteredOptions.map(option => (
            <li key={option.name} className="FilterCheckboxItem">
              <label>
                <input 
                  type="checkbox" 
                  checked={activeItems.includes(option.name)}
                  onChange={() => onToggle(option.name)}
                />
                <span className="FilterOptionName">{option.name}</span>
                <span className="FilterOptionCount">({option.count})</span>
              </label>
            </li>
          ))
        ) : (
          <li className="FilterNoResults">No results found for "{searchTerm}"</li>
        )}
      </ul>
    </div>
  );
};

const FilterPanel = ({ filterOptions, activeFilters, onToggleFilter, filterLogicMode, onToggleFilterLogicMode }) => {
  const categories = [
    { id: 'genre', title: 'Genre', options: filterOptions.genre, active: activeFilters.genre },
    { id: 'style', title: 'Style', options: filterOptions.style, active: activeFilters.style },
    { id: 'mood', title: 'Mood', options: filterOptions.mood, active: activeFilters.mood },
    { id: 'instrument', title: 'Instrument', options: filterOptions.instrument, active: activeFilters.instrument },
    { id: 'spectral', title: 'Spectral Features', options: filterOptions.spectral, active: activeFilters.spectral },
  ];

  return (
    <div className="FilterPanelOuterContainer">
      <div className="FilterPanelHeader">
        <button onClick={onToggleFilterLogicMode} className="FilterLogicButton">
          Match: {filterLogicMode === 'intersection' ? 'All Categories (AND)' : 'Any Tag (OR)'}
        </button>
      </div>
      <div className="FilterPanelContainer">
        {categories.map(cat => (
          <FilterCategory 
            key={cat.id}
            title={cat.title}
            options={cat.options}
            activeItems={cat.active || []}
            onToggle={(value) => onToggleFilter(cat.id, value)}
          />
        ))}
      </div>
    </div>
  );
};

export default FilterPanel; 